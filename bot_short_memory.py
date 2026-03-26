"""
bot_short_memory.py — Telegram-бот с короткой памятью (history buffer).

Стек:
  - aiogram 3.x  (async Telegram Bot API)
  - OpenAI-compatible SDK  (Chat Completions)
  - Три провайдера: Z.AI, ProxyAPI, GenAPI

Запуск:
  python bot_short_memory.py

Команды бота:
  /start   — приветствие
  /config  — выбрать провайдера, модель, температуру, макс. токены
  /new     — очистить историю диалога
  /info    — показать текущие настройки
"""

import asyncio
import logging
import os
from collections import defaultdict, deque
from typing import Deque, Dict, List, TypedDict

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums.parse_mode import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    CallbackQuery,
)
from dotenv import load_dotenv
from openai import OpenAI

# ─── Загрузка .env ────────────────────────────────────────────────────────────

load_dotenv()

# ─── Константы ────────────────────────────────────────────────────────────────

MAX_HISTORY = 10  # последних N сообщений (user + assistant) на пользователя

# ─── Провайдеры и модели (из ai_direct.py) ───────────────────────────────────

PROVIDERS: Dict[str, dict] = {
    "zai": {
        "name": "Z.AI",
        "api_key_env": "ZAI_API_KEY",
        "base_url": "https://api.z.ai/api/paas/v4/",
        "models": {
            "glm-4.7-flash": {"label": "GLM-4.7-Flash", "free": True,  "temp_range": (0.0, 1.0), "max_tokens": 4096},
            "glm-4.5-flash": {"label": "GLM-4.5-Flash", "free": True,  "temp_range": (0.0, 1.0), "max_tokens": 4096},
            "glm-4.7":       {"label": "GLM-4.7",       "free": False, "temp_range": (0.0, 1.0), "max_tokens": 8192},
            "glm-4.5":       {"label": "GLM-4.5",       "free": False, "temp_range": (0.0, 1.0), "max_tokens": 8192},
        },
    },
    "proxy": {
        "name": "ProxyAPI (OpenAI)",
        "api_key_env": "PROXY_API_KEY",
        "base_url": "https://api.proxyapi.ru/openai/v1",
        "models": {
            "gpt-4.1-nano": {"label": "GPT-4.1 Nano", "free": False, "temp_range": (0.0, 2.0), "max_tokens": 32768},
            "gpt-4.1-mini": {"label": "GPT-4.1 Mini", "free": False, "temp_range": (0.0, 2.0), "max_tokens": 32768},
            "gpt-4.1":      {"label": "GPT-4.1",      "free": False, "temp_range": (0.0, 2.0), "max_tokens": 32768},
            "gpt-4o-mini":  {"label": "GPT-4o Mini",  "free": False, "temp_range": (0.0, 2.0), "max_tokens": 16384},
            "gpt-4o":       {"label": "GPT-4o",       "free": False, "temp_range": (0.0, 2.0), "max_tokens": 16384},
        },
    },
    "gen": {
        "name": "GenAPI",
        "api_key_env": "GEN_API_KEY",
        "base_url": "https://proxy.gen-api.ru/v1",
        "models": {
            "gpt-4-1-mini":      {"label": "GPT-4.1 Mini",      "free": False, "temp_range": (0.0, 2.0), "max_tokens": 32768},
            "gpt-4-1":           {"label": "GPT-4.1",           "free": False, "temp_range": (0.0, 2.0), "max_tokens": 32768},
            "gpt-4o":            {"label": "GPT-4o",            "free": False, "temp_range": (0.0, 2.0), "max_tokens": 16384},
            "claude-sonnet-4-5": {"label": "Claude Sonnet 4.5", "free": False, "temp_range": (0.0, 1.0), "max_tokens": 8192},
            "gemini-2-5-flash":  {"label": "Gemini 2.5 Flash",  "free": False, "temp_range": (0.0, 2.0), "max_tokens": 8192},
            "deepseek-chat":     {"label": "DeepSeek Chat",     "free": False, "temp_range": (0.0, 2.0), "max_tokens": 8192},
            "deepseek-r1":       {"label": "DeepSeek R1",       "free": False, "temp_range": (0.0, 2.0), "max_tokens": 16000},
        },
    },
}

# Дефолтные настройки для новых пользователей
DEFAULT_PROVIDER = "proxy"
DEFAULT_MODEL    = "gpt-4o-mini"
DEFAULT_TEMP     = 0.7
DEFAULT_TOKENS   = 1000

# ─── Типы ─────────────────────────────────────────────────────────────────────

class ChatMessage(TypedDict):
    role: str       # "system" | "user" | "assistant"
    content: str


# Короткая память: user_id -> кольцевая очередь последних MAX_HISTORY сообщений
memory: Dict[int, Deque[ChatMessage]] = defaultdict(
    lambda: deque(maxlen=MAX_HISTORY)
)

# Настройки пользователя: user_id -> {provider, model, temperature, max_tokens}
user_settings: Dict[int, dict] = {}


def get_settings(user_id: int) -> dict:
    """Возвращает настройки пользователя, создавая дефолтные при первом обращении."""
    if user_id not in user_settings:
        user_settings[user_id] = {
            "provider": DEFAULT_PROVIDER,
            "model":    DEFAULT_MODEL,
            "temperature": DEFAULT_TEMP,
            "max_tokens":  DEFAULT_TOKENS,
        }
    return user_settings[user_id]


# ─── FSM: состояния для /config ───────────────────────────────────────────────

class ConfigStates(StatesGroup):
    waiting_temperature = State()
    waiting_max_tokens  = State()


# ─── Роутер ───────────────────────────────────────────────────────────────────

router = Router()


# ─── Клавиатуры ───────────────────────────────────────────────────────────────

def kb_providers() -> InlineKeyboardMarkup:
    """Клавиатура выбора провайдера."""
    buttons = [
        [InlineKeyboardButton(text=p["name"], callback_data=f"prov:{key}")]
        for key, p in PROVIDERS.items()
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def kb_models(provider_key: str) -> InlineKeyboardMarkup:
    """Клавиатура выбора модели для провайдера."""
    models = PROVIDERS[provider_key]["models"]
    buttons = [
        [InlineKeyboardButton(
            text=f"{'🆓 ' if m['free'] else ''}{m['label']}",
            callback_data=f"model:{provider_key}:{mid}"
        )]
        for mid, m in models.items()
    ]
    buttons.append([InlineKeyboardButton(text="◀ Назад", callback_data="back:providers")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def kb_tokens(provider_key: str, model_id: str) -> InlineKeyboardMarkup:
    """Клавиатура выбора макс. токенов."""
    max_possible = PROVIDERS[provider_key]["models"][model_id]["max_tokens"]
    # Предлагаем несколько вариантов, не превышающих лимит модели
    options = [500, 1000, 2000, 4000, 8000]
    options = [t for t in options if t <= max_possible]
    if max_possible not in options:
        options.append(max_possible)

    buttons = [
        [InlineKeyboardButton(text=str(t), callback_data=f"tokens:{provider_key}:{model_id}:{t}")]
        for t in options
    ]
    buttons.append([InlineKeyboardButton(
        text="✏️ Ввести вручную",
        callback_data=f"tokens_manual:{provider_key}:{model_id}"
    )])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


# ─── Хэндлеры команд ──────────────────────────────────────────────────────────

@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    """Приветствие и краткая инструкция."""
    uid = message.from_user.id
    s = get_settings(uid)
    prov_name = PROVIDERS[s["provider"]]["name"]
    model_label = PROVIDERS[s["provider"]]["models"].get(s["model"], {}).get("label", s["model"])

    await message.answer(
        "👋 Привет! Я бот с <b>короткой памятью</b>.\n\n"
        f"Текущие настройки:\n"
        f"  Провайдер: <b>{prov_name}</b>\n"
        f"  Модель: <b>{model_label}</b>\n"
        f"  Температура: <b>{s['temperature']}</b>\n"
        f"  Макс. токенов: <b>{s['max_tokens']}</b>\n\n"
        "Просто напиши мне — я запомню последние 10 реплик диалога.\n\n"
        "Команды:\n"
        "  /config — настроить провайдера и модель\n"
        "  /new    — очистить историю\n"
        "  /info   — текущие настройки",
        parse_mode=ParseMode.HTML,
    )


@router.message(Command("info"))
async def cmd_info(message: Message) -> None:
    """Показывает текущие настройки пользователя."""
    uid = message.from_user.id
    s = get_settings(uid)
    prov = PROVIDERS[s["provider"]]
    model_label = prov["models"].get(s["model"], {}).get("label", s["model"])
    history_len = len(memory[uid])

    await message.answer(
        f"⚙️ <b>Текущие настройки</b>\n\n"
        f"Провайдер:    <b>{prov['name']}</b>\n"
        f"Модель:       <b>{model_label}</b>\n"
        f"Температура:  <b>{s['temperature']}</b>\n"
        f"Макс. токенов: <b>{s['max_tokens']}</b>\n"
        f"История:      <b>{history_len}</b> сообщений в памяти",
        parse_mode=ParseMode.HTML,
    )


@router.message(Command("new"))
async def cmd_new(message: Message) -> None:
    """Очищает историю диалога пользователя."""
    uid = message.from_user.id
    memory[uid].clear()
    await message.answer("🗑 История очищена. Начинаем с чистого листа.")


@router.message(Command("config"))
async def cmd_config(message: Message, state: FSMContext) -> None:
    """Запускает настройку: выбор провайдера."""
    await state.clear()
    await message.answer("🔧 Выбери провайдера:", reply_markup=kb_providers())


# ─── Хэндлеры inline-кнопок (конфигурация) ───────────────────────────────────

@router.callback_query(F.data == "back:providers")
async def cb_back_providers(call: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await call.message.edit_text("🔧 Выбери провайдера:", reply_markup=kb_providers())
    await call.answer()


@router.callback_query(F.data.startswith("prov:"))
async def cb_provider(call: CallbackQuery) -> None:
    """Пользователь выбрал провайдера — показываем модели."""
    provider_key = call.data.split(":")[1]
    prov_name = PROVIDERS[provider_key]["name"]
    await call.message.edit_text(
        f"Провайдер: <b>{prov_name}</b>\nВыбери модель:",
        reply_markup=kb_models(provider_key),
        parse_mode=ParseMode.HTML,
    )
    await call.answer()


@router.callback_query(F.data.startswith("model:"))
async def cb_model(call: CallbackQuery, state: FSMContext) -> None:
    """Пользователь выбрал модель — запрашиваем температуру."""
    _, provider_key, model_id = call.data.split(":", 2)
    model_info = PROVIDERS[provider_key]["models"][model_id]
    lo, hi = model_info["temp_range"]

    # Сохраняем промежуточный выбор в FSM
    await state.update_data(provider=provider_key, model=model_id)
    await state.set_state(ConfigStates.waiting_temperature)

    await call.message.edit_text(
        f"Модель: <b>{model_info['label']}</b>\n\n"
        f"Введи температуру от {lo} до {hi} (например: <code>0.7</code>):",
        parse_mode=ParseMode.HTML,
    )
    await call.answer()


@router.message(ConfigStates.waiting_temperature)
async def fsm_temperature(message: Message, state: FSMContext) -> None:
    """Получаем температуру, переходим к выбору токенов."""
    data = await state.get_data()
    provider_key = data["provider"]
    model_id = data["model"]
    model_info = PROVIDERS[provider_key]["models"][model_id]
    lo, hi = model_info["temp_range"]

    try:
        temp = float(message.text.strip().replace(",", "."))
        temp = max(lo, min(hi, temp))
    except ValueError:
        await message.answer(f"Введи число от {lo} до {hi}:")
        return

    await state.update_data(temperature=temp)
    await state.set_state(ConfigStates.waiting_max_tokens)

    await message.answer(
        f"Температура: <b>{temp}</b>\n\nВыбери макс. количество токенов:",
        reply_markup=kb_tokens(provider_key, model_id),
        parse_mode=ParseMode.HTML,
    )


@router.callback_query(F.data.startswith("tokens:"))
async def cb_tokens(call: CallbackQuery, state: FSMContext) -> None:
    """Пользователь выбрал токены из кнопок — сохраняем настройки."""
    parts = call.data.split(":")
    provider_key = parts[1]
    model_id = parts[2]
    max_tokens = int(parts[3])

    data = await state.get_data()
    temperature = data.get("temperature", DEFAULT_TEMP)

    await _apply_settings(call, state, provider_key, model_id, temperature, max_tokens)


@router.callback_query(F.data.startswith("tokens_manual:"))
async def cb_tokens_manual(call: CallbackQuery, state: FSMContext) -> None:
    """Пользователь хочет ввести токены вручную."""
    _, provider_key, model_id = call.data.split(":", 2)
    max_possible = PROVIDERS[provider_key]["models"][model_id]["max_tokens"]

    await state.update_data(provider=provider_key, model=model_id)
    await state.set_state(ConfigStates.waiting_max_tokens)

    await call.message.edit_text(
        f"Введи количество токенов (1 – {max_possible}):",
        parse_mode=ParseMode.HTML,
    )
    await call.answer()


@router.message(ConfigStates.waiting_max_tokens)
async def fsm_max_tokens(message: Message, state: FSMContext) -> None:
    """Получаем токены вручную — сохраняем настройки."""
    data = await state.get_data()
    provider_key = data["provider"]
    model_id = data["model"]
    temperature = data.get("temperature", DEFAULT_TEMP)
    max_possible = PROVIDERS[provider_key]["models"][model_id]["max_tokens"]

    try:
        max_tokens = int(message.text.strip())
        max_tokens = max(1, min(max_possible, max_tokens))
    except ValueError:
        await message.answer(f"Введи целое число от 1 до {max_possible}:")
        return

    await _apply_settings(message, state, provider_key, model_id, temperature, max_tokens)


async def _apply_settings(
    event,  # Message или CallbackQuery
    state: FSMContext,
    provider_key: str,
    model_id: str,
    temperature: float,
    max_tokens: int,
) -> None:
    """Сохраняет финальные настройки пользователя и завершает FSM."""
    # Определяем user_id из Message или CallbackQuery
    if isinstance(event, CallbackQuery):
        uid = event.from_user.id
        send = event.message.answer
        await event.answer()
    else:
        uid = event.from_user.id
        send = event.answer

    user_settings[uid] = {
        "provider":    provider_key,
        "model":       model_id,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    await state.clear()

    prov_name    = PROVIDERS[provider_key]["name"]
    model_label  = PROVIDERS[provider_key]["models"][model_id]["label"]

    await send(
        f"✅ Настройки сохранены!\n\n"
        f"Провайдер:    <b>{prov_name}</b>\n"
        f"Модель:       <b>{model_label}</b>\n"
        f"Температура:  <b>{temperature}</b>\n"
        f"Макс. токенов: <b>{max_tokens}</b>\n\n"
        "Теперь можешь писать — я готов к диалогу.",
        parse_mode=ParseMode.HTML,
    )


# ─── Основной хэндлер: текстовые сообщения ───────────────────────────────────

@router.message(F.text)
async def on_text(message: Message) -> None:
    """
    Обрабатывает входящее сообщение:
    1. Добавляет его в короткую память пользователя.
    2. Формирует payload (история + текущее сообщение) для Chat Completions.
    3. Отправляет запрос к выбранному провайдеру.
    4. Возвращает ответ пользователю и добавляет его в память.
    """
    uid = message.from_user.id
    user_text = message.text.strip()
    s = get_settings(uid)

    provider_key = s["provider"]
    model_id     = s["model"]
    temperature  = s["temperature"]
    max_tokens   = s["max_tokens"]

    provider = PROVIDERS[provider_key]
    api_key  = os.getenv(provider["api_key_env"])

    if not api_key:
        await message.answer(
            f"⚠️ Не найден API-ключ для провайдера <b>{provider['name']}</b>.\n"
            f"Проверь переменную <code>{provider['api_key_env']}</code> в .env",
            parse_mode=ParseMode.HTML,
        )
        return

    # Добавляем сообщение пользователя в короткую память
    memory[uid].append({"role": "user", "content": user_text})

    # Формируем payload: системный промпт + история + текущее сообщение
    system_msg: ChatMessage = {
        "role": "system",
        "content": "Ты — дружелюбный и лаконичный помощник. Отвечай по делу.",
    }
    messages: List[ChatMessage] = [system_msg, *list(memory[uid])]

    # Индикатор "печатает..."
    await message.bot.send_chat_action(message.chat.id, "typing")

    try:
        client = OpenAI(api_key=api_key, base_url=provider["base_url"])
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        reply = response.choices[0].message.content or ""

        # Обрабатываем reasoning_content (DeepSeek R1, GLM-Z)
        if not reply:
            rc = getattr(response.choices[0].message, "reasoning_content", None)
            if rc:
                reply = rc

        if not reply:
            reply = "⚠️ Модель вернула пустой ответ."

    except Exception as e:
        logging.exception("Ошибка при запросе к %s", provider["name"])
        await message.answer(f"❌ Ошибка: <code>{e}</code>", parse_mode=ParseMode.HTML)
        # Убираем последнее сообщение из памяти, т.к. ответа не было
        memory[uid].pop()
        return

    # Отправляем ответ пользователю
    await message.answer(reply)

    # Добавляем ответ ассистента в короткую память
    memory[uid].append({"role": "assistant", "content": reply})

    # Логируем использование токенов
    if response.usage:
        u = response.usage
        logging.info(
            "user=%d provider=%s model=%s tokens: in=%d out=%d total=%d",
            uid, provider_key, model_id,
            u.prompt_tokens, u.completion_tokens, u.total_tokens,
        )


# ─── Точка входа ──────────────────────────────────────────────────────────────

async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    bot_token = os.getenv("BOT_TOKEN")
    if not bot_token:
        raise RuntimeError("BOT_TOKEN не найден в .env")

    bot = Bot(token=bot_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp  = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)

    logging.info("Бот запущен. Ожидаю сообщения...")
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
