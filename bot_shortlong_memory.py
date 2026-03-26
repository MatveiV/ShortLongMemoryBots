"""
bot_shortlong_memory.py — Telegram-бот с короткой И долгой памятью.

Короткая память: последние 10 сообщений диалога (deque в RAM).
Долгая память:  документы (PDF/TXT/DOCX) → эмбеддинги → ChromaDB (persistent).

При каждом вопросе бот:
  1. Ищет релевантные фрагменты в ChromaDB (долгая память).
  2. Добавляет историю диалога (короткая память).
  3. Передаёт всё модели и возвращает ответ.

Стек: aiogram 3.x · OpenAI-compatible SDK · ChromaDB

Команды:
  /start   — приветствие
  /config  — провайдер, модель, температура, токены, модель эмбеддингов
  /new     — очистить короткую память (историю диалога)
  /docs    — список загруженных документов
  /clear   — удалить все документы из ChromaDB
  /info    — текущие настройки
"""

import asyncio
import logging
import os
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, TypedDict

import aiohttp
import chromadb
from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums.parse_mode import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from dotenv import load_dotenv
from openai import OpenAI

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import docx as python_docx
except ImportError:
    python_docx = None

load_dotenv()

# ─── Константы ────────────────────────────────────────────────────────────────

PERSIST_DIR    = "./memory"
UPLOADS_DIR    = "./uploads"
CHUNK_SIZE     = 500
CHUNK_OVERLAP  = 50
TOP_K          = 5
MAX_HISTORY    = 10   # размер короткой памяти (user + assistant сообщений)


# ─── Провайдеры ───────────────────────────────────────────────────────────────

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
        "embed_models": {
            "embedding-3": {"label": "Embedding-3 (Z.AI)"},
            "embedding-2": {"label": "Embedding-2 (Z.AI)"},
        },
        "default_embed": "embedding-3",
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
        "embed_models": {
            "text-embedding-3-small": {"label": "text-embedding-3-small"},
            "text-embedding-3-large": {"label": "text-embedding-3-large"},
            "text-embedding-ada-002": {"label": "text-embedding-ada-002"},
        },
        "default_embed": "text-embedding-3-small",
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
        "embed_models": {
            "text-embedding-3-small": {"label": "text-embedding-3-small"},
            "text-embedding-3-large": {"label": "text-embedding-3-large"},
            "text-embedding-ada-002": {"label": "text-embedding-ada-002"},
        },
        "default_embed": "text-embedding-3-small",
    },
}

DEFAULT_PROVIDER = "proxy"
DEFAULT_MODEL    = "gpt-4o-mini"
DEFAULT_TEMP     = 0.5
DEFAULT_TOKENS   = 1000

# Режимы работы бота
MODE_SHORT    = "short"     # только короткая память
MODE_LONG     = "long"      # только долгая память (RAG)
MODE_COMBINED = "combined"  # короткая + долгая память

# ─── Типы ─────────────────────────────────────────────────────────────────────

class ChatMessage(TypedDict):
    role: str
    content: str

# ─── Состояние пользователей ──────────────────────────────────────────────────

# Настройки: user_id -> {provider, model, temperature, max_tokens, embed_model, mode}
user_settings: Dict[int, dict] = {}

# Короткая память: user_id -> deque последних MAX_HISTORY сообщений
short_memory: Dict[int, Deque[ChatMessage]] = defaultdict(
    lambda: deque(maxlen=MAX_HISTORY)
)

# Документы сессии: user_id -> [{doc_id, filename, chunks}]
user_docs: Dict[int, List[dict]] = {}


def get_settings(user_id: int) -> dict:
    if user_id not in user_settings:
        pkey = DEFAULT_PROVIDER
        user_settings[user_id] = {
            "provider":    pkey,
            "model":       DEFAULT_MODEL,
            "temperature": DEFAULT_TEMP,
            "max_tokens":  DEFAULT_TOKENS,
            "embed_model": PROVIDERS[pkey]["default_embed"],
            "mode":        MODE_COMBINED,
        }
    return user_settings[user_id]


# ─── FSM ──────────────────────────────────────────────────────────────────────

class ConfigStates(StatesGroup):
    waiting_temperature = State()
    waiting_max_tokens  = State()
    waiting_embed_model = State()


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def ensure_dirs() -> None:
    Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    Path(UPLOADS_DIR).mkdir(parents=True, exist_ok=True)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Разбивает текст на пересекающиеся чанки."""
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def load_document(file_path: str) -> List[str]:
    """Извлекает текст из PDF/TXT/DOCX и возвращает список чанков."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore")
    elif suffix == ".pdf":
        if PdfReader is None:
            raise RuntimeError("Для PDF установите: pip install pypdf")
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix in (".docx", ".doc"):
        if python_docx is None:
            raise RuntimeError("Для DOCX установите: pip install python-docx")
        doc = python_docx.Document(str(path))
        text = "\n".join(p.text for p in doc.paragraphs)
    else:
        raise RuntimeError("Поддерживаются только PDF, TXT, DOCX")

    return chunk_text(text)


def get_collection() -> chromadb.Collection:
    """Возвращает ChromaDB коллекцию (cosine-метрика)."""
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_or_create_collection(
        name="long_memory",
        metadata={"hnsw:space": "cosine"},
    )


def embed_chunks(openai_client: OpenAI, user_id: int, doc_id: str,
                 chunks: List[str], embed_model: str) -> int:
    """Вычисляет эмбеддинги и сохраняет чанки в ChromaDB (долгая память)."""
    if not chunks:
        return 0
    resp = openai_client.embeddings.create(model=embed_model, input=chunks)
    embeddings = [d.embedding for d in resp.data]
    collection = get_collection()
    ids       = [f"{user_id}:{doc_id}:{i}" for i in range(len(chunks))]
    metadatas = [{"user_id": str(user_id), "doc_id": doc_id, "chunk_index": i}
                 for i in range(len(chunks))]
    collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=chunks)
    return len(chunks)


def retrieve_context(openai_client: OpenAI, user_id: int, question: str,
                     embed_model: str, top_k: int = TOP_K,
                     doc_id: Optional[str] = None) -> List[str]:
    """Ищет top_k релевантных фрагментов в ChromaDB по вопросу (долгая память)."""
    q_emb = openai_client.embeddings.create(
        model=embed_model, input=[question]
    ).data[0].embedding

    where: dict = {"user_id": str(user_id)}
    if doc_id:
        where["doc_id"] = doc_id

    collection = get_collection()
    existing = collection.get(where={"user_id": str(user_id)}, limit=1)
    if not existing["ids"]:
        return []

    result = collection.query(query_embeddings=[q_emb], n_results=top_k, where=where)
    docs = result.get("documents") or []
    return docs[0] if docs else []


async def answer_with_memory(
    openai_client: OpenAI,
    user_id: int,
    user_text: str,
    model_id: str,
    temperature: float,
    max_tokens: int,
    embed_model: str,
) -> str:
    """
    Объединённый RAG + короткая память:
    1. Ищет релевантные фрагменты в ChromaDB (долгая память).
    2. Берёт историю диалога (короткая память).
    3. Передаёт всё модели и возвращает ответ.
    """
    # Долгая память: поиск по документам
    context_chunks: List[str] = []
    try:
        context_chunks = await asyncio.to_thread(
            retrieve_context, openai_client, user_id, user_text, embed_model, TOP_K
        )
    except Exception:
        logging.exception("RAG retrieve failed")

    # Системный промпт: приоритет документу, но не игнорировать историю
    system_content = (
        "Ты — умный помощник с двумя видами памяти.\n"
        "1. Долгая память: фрагменты загруженных документов (если есть).\n"
        "2. Короткая память: история текущего диалога.\n"
        "Если в контексте документа есть ответ — опирайся прежде всего на него. "
        "Если контекста нет или недостаточно — используй историю и общие знания. "
        "Не выдумывай факты о документе, которых нет в контексте."
    )

    # Формируем сообщение пользователя: добавляем контекст документа если найден
    if context_chunks:
        context_text = "\n\n".join(
            f"[Фрагмент {i+1}]\n{c}" for i, c in enumerate(context_chunks)
        )
        user_content = f"Контекст из документа:\n{context_text}\n\nВопрос: {user_text}"
    else:
        user_content = user_text

    # Короткая память: история диалога
    history = list(short_memory[user_id])

    messages: List[ChatMessage] = [
        {"role": "system", "content": system_content},
        *history,
        {"role": "user", "content": user_content},
    ]

    completion = await asyncio.to_thread(
        openai_client.chat.completions.create,
        model=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    reply = completion.choices[0].message.content or ""
    # Обработка reasoning_content (DeepSeek R1, GLM-Z)
    if not reply:
        rc = getattr(completion.choices[0].message, "reasoning_content", None)
        if rc:
            reply = rc

    return reply or "⚠️ Модель вернула пустой ответ."


# ─── Клавиатуры ───────────────────────────────────────────────────────────────

def kb_providers() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=p["name"], callback_data=f"prov:{key}")]
        for key, p in PROVIDERS.items()
    ])


def kb_models(provider_key: str) -> InlineKeyboardMarkup:
    models = PROVIDERS[provider_key]["models"]
    buttons = [
        [InlineKeyboardButton(
            text=f"{'🆓 ' if m['free'] else ''}{m['label']}",
            callback_data=f"model:{provider_key}:{mid}",
        )]
        for mid, m in models.items()
    ]
    buttons.append([InlineKeyboardButton(text="◀ Назад", callback_data="back:providers")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def kb_tokens(provider_key: str, model_id: str) -> InlineKeyboardMarkup:
    max_possible = PROVIDERS[provider_key]["models"][model_id]["max_tokens"]
    options = [t for t in [500, 1000, 2000, 4000, 8000] if t <= max_possible]
    if max_possible not in options:
        options.append(max_possible)
    buttons = [
        [InlineKeyboardButton(text=str(t), callback_data=f"tokens:{provider_key}:{model_id}:{t}")]
        for t in options
    ]
    buttons.append([InlineKeyboardButton(
        text="✏️ Ввести вручную",
        callback_data=f"tokens_manual:{provider_key}:{model_id}",
    )])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def kb_embed_models(provider_key: str) -> InlineKeyboardMarkup:
    embed_models = PROVIDERS[provider_key]["embed_models"]
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=m["label"], callback_data=f"embed:{provider_key}:{mid}")]
        for mid, m in embed_models.items()
    ])


def kb_mode() -> InlineKeyboardMarkup:
    """Клавиатура выбора режима памяти."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="💬 Короткая память",          callback_data=f"mode:{MODE_SHORT}")],
        [InlineKeyboardButton(text="📚 Долгая память (документы)", callback_data=f"mode:{MODE_LONG}")],
        [InlineKeyboardButton(text="🧠 Короткая + Долгая память",  callback_data=f"mode:{MODE_COMBINED}")],
    ])


# ─── Роутер ───────────────────────────────────────────────────────────────────

router = Router()


# ─── Команды ──────────────────────────────────────────────────────────────────

@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        "👋 Привет! Я универсальный AI-ассистент.\n\n"
        "Выбери режим работы:",
        reply_markup=kb_mode(),
    )


MODE_DESCRIPTIONS = {
    MODE_SHORT: (
        "💬 <b>Режим: Короткая память</b>\n\n"
        "Помню последние 10 реплик диалога.\n"
        "Документы не используются.\n\n"
        "Просто пиши — я отвечу с учётом истории разговора."
    ),
    MODE_LONG: (
        "📚 <b>Режим: Долгая память (документы)</b>\n\n"
        "Отвечаю строго по загруженным документам.\n"
        "История диалога не сохраняется.\n\n"
        "Загрузи PDF/TXT/DOCX, затем задавай вопросы."
    ),
    MODE_COMBINED: (
        "🧠 <b>Режим: Короткая + Долгая память</b>\n\n"
        "Помню историю диалога И использую загруженные документы.\n"
        "При ответе сначала ищу в документах, затем дополняю историей.\n\n"
        "Загрузи документ или просто пиши — работает в обоих случаях."
    ),
}


@router.callback_query(F.data.startswith("mode:"))
async def cb_mode(call: CallbackQuery) -> None:
    """Пользователь выбрал режим памяти."""
    mode = call.data.split(":")[1]
    uid = call.from_user.id
    s = get_settings(uid)
    s["mode"] = mode
    prov = PROVIDERS[s["provider"]]
    model_label = prov["models"].get(s["model"], {}).get("label", s["model"])

    commands = {
        MODE_SHORT:    "/config — настройки  |  /new — очистить историю  |  /info — инфо",
        MODE_LONG:     "/config — настройки  |  /docs — документы  |  /clear — удалить базу  |  /info — инфо",
        MODE_COMBINED: "/config — настройки  |  /new — история  |  /docs — документы  |  /clear — база  |  /info — инфо",
    }

    await call.message.edit_text(
        f"{MODE_DESCRIPTIONS[mode]}\n\n"
        f"Провайдер: <b>{prov['name']}</b> | Модель: <b>{model_label}</b>\n\n"
        f"{commands[mode]}",
    )
    await call.answer()


@router.message(Command("info"))
async def cmd_info(message: Message) -> None:
    uid = message.from_user.id
    s = get_settings(uid)
    prov = PROVIDERS[s["provider"]]
    model_label = prov["models"].get(s["model"], {}).get("label", s["model"])
    history_len = len(short_memory[uid])
    docs_count  = len(user_docs.get(uid, []))
    mode_labels = {MODE_SHORT: "💬 Короткая", MODE_LONG: "📚 Долгая", MODE_COMBINED: "🧠 Короткая + Долгая"}
    await message.answer(
        f"⚙️ <b>Настройки</b>\n\n"
        f"Режим:         <b>{mode_labels.get(s['mode'], s['mode'])}</b>\n"
        f"Провайдер:     <b>{prov['name']}</b>\n"
        f"Модель:        <b>{model_label}</b>\n"
        f"Температура:   <b>{s['temperature']}</b>\n"
        f"Макс. токенов: <b>{s['max_tokens']}</b>\n"
        f"Эмбеддинги:    <b>{s['embed_model']}</b>\n\n"
        f"Короткая память: <b>{history_len}</b> сообщений\n"
        f"Документов:      <b>{docs_count}</b>",
    )


@router.message(Command("new"))
async def cmd_new(message: Message) -> None:
    """Очищает короткую память (историю диалога)."""
    uid = message.from_user.id
    short_memory[uid].clear()
    await message.answer("🗑 История диалога очищена. Документы сохранены.")


@router.message(Command("docs"))
async def cmd_docs(message: Message) -> None:
    uid = message.from_user.id
    docs = user_docs.get(uid, [])
    if not docs:
        await message.answer("У тебя пока нет загруженных документов.")
        return
    lines = [f"{i+1}. {d['filename']} ({d['chunks']} чанков)" for i, d in enumerate(docs)]
    await message.answer("📄 <b>Твои документы:</b>\n\n" + "\n".join(lines))


@router.message(Command("clear"))
async def cmd_clear(message: Message) -> None:
    """Удаляет все документы пользователя из ChromaDB."""
    uid = message.from_user.id
    collection = get_collection()
    existing = collection.get(where={"user_id": str(uid)})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
    user_docs.pop(uid, None)
    await message.answer("🗑 Все документы удалены из базы. История диалога сохранена.")


@router.message(Command("config"))
async def cmd_config(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("🔧 Выбери провайдера:", reply_markup=kb_providers())


# ─── Inline-кнопки конфигурации ───────────────────────────────────────────────

@router.callback_query(F.data == "back:providers")
async def cb_back(call: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await call.message.edit_text("🔧 Выбери провайдера:", reply_markup=kb_providers())
    await call.answer()


@router.callback_query(F.data.startswith("prov:"))
async def cb_provider(call: CallbackQuery) -> None:
    provider_key = call.data.split(":")[1]
    await call.message.edit_text(
        f"Провайдер: <b>{PROVIDERS[provider_key]['name']}</b>\nВыбери модель:",
        reply_markup=kb_models(provider_key),
    )
    await call.answer()


@router.callback_query(F.data.startswith("model:"))
async def cb_model(call: CallbackQuery, state: FSMContext) -> None:
    _, provider_key, model_id = call.data.split(":", 2)
    model_info = PROVIDERS[provider_key]["models"][model_id]
    lo, hi = model_info["temp_range"]
    await state.update_data(provider=provider_key, model=model_id)
    await state.set_state(ConfigStates.waiting_temperature)
    await call.message.edit_text(
        f"Модель: <b>{model_info['label']}</b>\n\n"
        f"Введи температуру от {lo} до {hi} (например: <code>0.5</code>):",
    )
    await call.answer()


@router.message(ConfigStates.waiting_temperature)
async def fsm_temperature(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    provider_key, model_id = data["provider"], data["model"]
    lo, hi = PROVIDERS[provider_key]["models"][model_id]["temp_range"]
    try:
        temp = max(lo, min(hi, float(message.text.strip().replace(",", "."))))
    except ValueError:
        await message.answer(f"Введи число от {lo} до {hi}:")
        return
    await state.update_data(temperature=temp)
    await state.set_state(ConfigStates.waiting_max_tokens)
    await message.answer(
        f"Температура: <b>{temp}</b>\n\nВыбери макс. токенов:",
        reply_markup=kb_tokens(provider_key, model_id),
    )


@router.callback_query(F.data.startswith("tokens:"))
async def cb_tokens(call: CallbackQuery, state: FSMContext) -> None:
    parts = call.data.split(":")
    provider_key, model_id, max_tokens = parts[1], parts[2], int(parts[3])
    data = await state.get_data()
    await _ask_embed_model(call, state, provider_key, model_id,
                           data.get("temperature", DEFAULT_TEMP), max_tokens)


@router.callback_query(F.data.startswith("tokens_manual:"))
async def cb_tokens_manual(call: CallbackQuery, state: FSMContext) -> None:
    _, provider_key, model_id = call.data.split(":", 2)
    max_possible = PROVIDERS[provider_key]["models"][model_id]["max_tokens"]
    await state.update_data(provider=provider_key, model=model_id)
    await state.set_state(ConfigStates.waiting_max_tokens)
    await call.message.edit_text(f"Введи количество токенов (1 – {max_possible}):")
    await call.answer()


@router.message(ConfigStates.waiting_max_tokens)
async def fsm_max_tokens(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    provider_key, model_id = data["provider"], data["model"]
    max_possible = PROVIDERS[provider_key]["models"][model_id]["max_tokens"]
    try:
        max_tokens = max(1, min(max_possible, int(message.text.strip())))
    except ValueError:
        await message.answer(f"Введи целое число от 1 до {max_possible}:")
        return
    await _ask_embed_model(message, state, provider_key, model_id,
                           data.get("temperature", DEFAULT_TEMP), max_tokens)


async def _ask_embed_model(event, state, provider_key, model_id, temperature, max_tokens) -> None:
    """Переходим к выбору модели эмбеддингов."""
    await state.update_data(provider=provider_key, model=model_id,
                            temperature=temperature, max_tokens=max_tokens)
    await state.set_state(ConfigStates.waiting_embed_model)
    send = event.message.answer if isinstance(event, CallbackQuery) else event.answer
    if isinstance(event, CallbackQuery):
        await event.answer()
    await send(
        f"Макс. токенов: <b>{max_tokens}</b>\n\nВыбери модель эмбеддингов:",
        reply_markup=kb_embed_models(provider_key),
    )


@router.callback_query(F.data.startswith("embed:"))
async def cb_embed(call: CallbackQuery, state: FSMContext) -> None:
    """Финальное сохранение всех настроек."""
    _, provider_key, embed_model_id = call.data.split(":", 2)
    data = await state.get_data()
    uid = call.from_user.id

    user_settings[uid] = {
        "provider":    provider_key,
        "model":       data["model"],
        "temperature": data["temperature"],
        "max_tokens":  data["max_tokens"],
        "embed_model": embed_model_id,
    }
    await state.clear()

    prov        = PROVIDERS[provider_key]
    model_label = prov["models"][data["model"]]["label"]
    embed_label = prov["embed_models"][embed_model_id]["label"]

    await call.message.answer(
        f"✅ Настройки сохранены!\n\n"
        f"Провайдер:     <b>{prov['name']}</b>\n"
        f"Модель:        <b>{model_label}</b>\n"
        f"Температура:   <b>{data['temperature']}</b>\n"
        f"Макс. токенов: <b>{data['max_tokens']}</b>\n"
        f"Эмбеддинги:    <b>{embed_label}</b>",
    )
    await call.answer()


# ─── Загрузка документа (долгая память) ──────────────────────────────────────

@router.message(F.document)
async def on_document(message: Message, bot: Bot) -> None:
    """
    Скачивает файл, парсит текст, создаёт эмбеддинги и сохраняет в ChromaDB.
    """
    uid = message.from_user.id
    tg_doc = message.document
    s = get_settings(uid)
    provider = PROVIDERS[s["provider"]]

    api_key = os.getenv(provider["api_key_env"])
    if not api_key:
        await message.answer(f"⚠️ Не найден ключ {provider['api_key_env']} в .env")
        return

    client = OpenAI(api_key=api_key, base_url=provider["base_url"])

    # Скачиваем файл
    ensure_dirs()
    user_dir = Path(UPLOADS_DIR) / str(uid)
    user_dir.mkdir(parents=True, exist_ok=True)
    filename  = tg_doc.file_name or f"file_{uuid.uuid4().hex}"
    save_path = user_dir / filename

    try:
        file_info = await bot.get_file(tg_doc.file_id)
        file_url  = f"https://api.telegram.org/file/bot{bot.token}/{file_info.file_path}"
        async with aiohttp.ClientSession() as session:
            async with session.get(file_url) as resp:
                resp.raise_for_status()
                save_path.write_bytes(await resp.read())
    except Exception as e:
        logging.exception("Ошибка скачивания файла")
        await message.answer(f"❌ Не удалось скачать файл: {e}")
        return

    await message.answer("⏳ Обрабатываю документ...")

    # Парсим и чанкуем
    try:
        chunks = load_document(str(save_path))
    except Exception as e:
        await message.answer(f"❌ {e}")
        return

    if not chunks:
        await message.answer("⚠️ Не удалось извлечь текст из документа.")
        return

    # Сохраняем эмбеддинги в ChromaDB (долгая память)
    doc_id = uuid.uuid4().hex
    try:
        count = await asyncio.to_thread(
            embed_chunks, client, uid, doc_id, chunks, s["embed_model"]
        )
    except Exception as e:
        logging.exception("Ошибка эмбеддинга/ChromaDB")
        await message.answer(f"❌ Ошибка при индексации: {e}")
        return

    user_docs.setdefault(uid, []).append(
        {"doc_id": doc_id, "filename": filename, "chunks": count}
    )

    await message.answer(
        f"✅ Документ <b>{filename}</b> проиндексирован.\n"
        f"Чанков: <b>{count}</b>\n\n"
        "Теперь задай вопрос — я использую и документ, и историю диалога."
    )


# ─── Основной хэндлер: текст (режим-зависимый) ───────────────────────────────

@router.message(F.text)
async def on_text(message: Message) -> None:
    """
    Маршрутизирует запрос в зависимости от выбранного режима:
    - MODE_SHORT:    только короткая память (история диалога)
    - MODE_LONG:     только долгая память (RAG по документам)
    - MODE_COMBINED: короткая + долгая память
    """
    uid = message.from_user.id
    user_text = message.text.strip()
    s = get_settings(uid)
    mode = s["mode"]
    provider = PROVIDERS[s["provider"]]

    api_key = os.getenv(provider["api_key_env"])
    if not api_key:
        await message.answer(f"⚠️ Не найден ключ {provider['api_key_env']} в .env")
        return

    client = OpenAI(api_key=api_key, base_url=provider["base_url"])
    await message.bot.send_chat_action(message.chat.id, "typing")

    try:
        if mode == MODE_SHORT:
            # ── Только короткая память ────────────────────────────────────
            short_memory[uid].append({"role": "user", "content": user_text})
            system_msg: ChatMessage = {
                "role": "system",
                "content": "Ты — дружелюбный и лаконичный помощник. Отвечай по делу.",
            }
            messages = [system_msg, *list(short_memory[uid])]
            completion = await asyncio.to_thread(
                client.chat.completions.create,
                model=s["model"], messages=messages,
                temperature=s["temperature"], max_tokens=s["max_tokens"],
            )
            reply = completion.choices[0].message.content or ""
            if not reply:
                reply = getattr(completion.choices[0].message, "reasoning_content", "") or "⚠️ Пустой ответ."
            await message.answer(reply)
            short_memory[uid].append({"role": "assistant", "content": reply})

        elif mode == MODE_LONG:
            # ── Только долгая память (RAG) ────────────────────────────────
            context_chunks = await asyncio.to_thread(
                retrieve_context, client, uid, user_text, s["embed_model"], TOP_K
            )
            if not context_chunks:
                await message.answer("📭 Контекст не найден. Загрузи документ и попробуй снова.")
                return
            context_text = "\n\n".join(f"[Фрагмент {i+1}]\n{c}" for i, c in enumerate(context_chunks))
            messages = [
                {"role": "system", "content": "Отвечай ТОЛЬКО на основе контекста. Если ответа нет — скажи об этом."},
                {"role": "user", "content": f"Контекст:\n{context_text}\n\nВопрос: {user_text}"},
            ]
            completion = await asyncio.to_thread(
                client.chat.completions.create,
                model=s["model"], messages=messages,
                temperature=s["temperature"], max_tokens=s["max_tokens"],
            )
            reply = completion.choices[0].message.content or "⚠️ Пустой ответ."
            await message.answer(reply)

        else:
            # ── Короткая + долгая память (combined) ───────────────────────
            short_memory[uid].append({"role": "user", "content": user_text})
            reply = await answer_with_memory(
                client, uid, user_text,
                s["model"], s["temperature"], s["max_tokens"], s["embed_model"],
            )
            await message.answer(reply)
            short_memory[uid].append({"role": "assistant", "content": reply})

    except Exception as e:
        logging.exception("Ошибка генерации ответа")
        await message.answer(f"❌ Ошибка: <code>{e}</code>")
        if mode in (MODE_SHORT, MODE_COMBINED) and short_memory[uid]:
            short_memory[uid].pop()


# ─── Точка входа ──────────────────────────────────────────────────────────────

async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ensure_dirs()

    bot_token = os.getenv("BOT_TOKEN")
    if not bot_token:
        raise RuntimeError("BOT_TOKEN не найден в .env")

    bot = Bot(token=bot_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp  = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)

    logging.info("Бот (короткая + долгая память) запущен.")
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
