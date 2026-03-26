"""
bot_long_memory.py — Telegram-бот с долгой памятью (RAG через ChromaDB).

Стек:
  - aiogram 3.x
  - OpenAI-compatible SDK (Chat Completions + Embeddings)
  - ChromaDB (persistent local vector store)
  - Три провайдера: Z.AI, ProxyAPI, GenAPI (как в ai_direct.py)

Запуск:
  python bot_long_memory.py

Команды:
  /start   — приветствие
  /config  — выбрать провайдера, модель, температуру, токены
  /docs    — список загруженных документов
  /clear   — удалить все документы пользователя из базы
  /info    — текущие настройки
"""

import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional

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

# Опциональные парсеры документов
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import docx as python_docx
except ImportError:
    python_docx = None

# ─── Загрузка .env ────────────────────────────────────────────────────────────

load_dotenv()

# ─── Константы ────────────────────────────────────────────────────────────────

PERSIST_DIR  = "./memory"   # ChromaDB persistent storage
UPLOADS_DIR  = "./uploads"  # загруженные файлы
CHUNK_SIZE   = 500
CHUNK_OVERLAP = 50
TOP_K        = 5            # кол-во релевантных фрагментов для контекста


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
        # Модели эмбеддингов Z.AI (Zhipu)
        "embed_models": {
            "embedding-3":   {"label": "Embedding-3 (Z.AI)"},
            "embedding-2":   {"label": "Embedding-2 (Z.AI)"},
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
        # Модели эмбеддингов OpenAI через ProxyAPI
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
        # Модели эмбеддингов OpenAI через GenAPI
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
DEFAULT_TEMP     = 0.2
DEFAULT_TOKENS   = 1000
DEFAULT_EMBED    = "text-embedding-3-small"

# Настройки: user_id -> {provider, model, temperature, max_tokens, embed_model}
user_settings: Dict[int, dict] = {}

# Документы пользователя: user_id -> [{doc_id, filename, chunks}]
user_docs: Dict[int, List[dict]] = {}


def get_settings(user_id: int) -> dict:
    if user_id not in user_settings:
        provider_key = DEFAULT_PROVIDER
        user_settings[user_id] = {
            "provider":    provider_key,
            "model":       DEFAULT_MODEL,
            "temperature": DEFAULT_TEMP,
            "max_tokens":  DEFAULT_TOKENS,
            "embed_model": PROVIDERS[provider_key]["default_embed"],
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
    """Разбивает текст на пересекающиеся чанки заданного размера."""
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
        start = end - overlap
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
    """Возвращает (или создаёт) ChromaDB коллекцию с cosine-метрикой."""
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_or_create_collection(
        name="long_memory",
        metadata={"hnsw:space": "cosine"},
    )


def embed_chunks(openai_client: OpenAI, user_id: int, doc_id: str, chunks: List[str], embed_model: str) -> int:
    """Вычисляет эмбеддинги чанков и сохраняет их в ChromaDB."""
    if not chunks:
        return 0

    # Получаем эмбеддинги батчем
    resp = openai_client.embeddings.create(model=embed_model, input=chunks)
    embeddings = [d.embedding for d in resp.data]

    collection = get_collection()
    ids       = [f"{user_id}:{doc_id}:{i}" for i in range(len(chunks))]
    metadatas = [{"user_id": str(user_id), "doc_id": doc_id, "chunk_index": i}
                 for i in range(len(chunks))]

    collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=chunks)
    return len(chunks)


def retrieve_context(
    openai_client: OpenAI,
    user_id: int,
    question: str,
    embed_model: str,
    top_k: int = TOP_K,
    doc_id: Optional[str] = None,
) -> List[str]:
    """Ищет top_k релевантных фрагментов в ChromaDB по вопросу пользователя."""
    q_emb = openai_client.embeddings.create(model=embed_model, input=[question]).data[0].embedding

    where: dict = {"user_id": str(user_id)}
    if doc_id:
        where["doc_id"] = doc_id

    collection = get_collection()

    # Проверяем, есть ли вообще документы пользователя
    existing = collection.get(where={"user_id": str(user_id)}, limit=1)
    if not existing["ids"]:
        return []

    result = collection.query(query_embeddings=[q_emb], n_results=top_k, where=where)
    docs = result.get("documents") or []
    return docs[0] if docs else []


async def answer_question(
    openai_client: OpenAI,
    context_chunks: List[str],
    question: str,
    model_id: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Формирует ответ через Chat Completions строго на основе найденного контекста."""
    context_text = "\n\n".join(
        f"Фрагмент {i+1}:\n{c}" for i, c in enumerate(context_chunks)
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Ты — помощник по документам. Отвечай ТОЛЬКО на основе предоставленного контекста. "
                "Если информации недостаточно — ответь: 'Не нашёл ответа в документе.'"
            ),
        },
        {
            "role": "user",
            "content": f"Контекст:\n{context_text}\n\nВопрос: {question}",
        },
    ]
    completion = await asyncio.to_thread(
        openai_client.chat.completions.create,
        model=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content or ""


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
    """Клавиатура выбора модели эмбеддингов для провайдера."""
    embed_models = PROVIDERS[provider_key]["embed_models"]
    buttons = [
        [InlineKeyboardButton(text=m["label"], callback_data=f"embed:{provider_key}:{mid}")]
        for mid, m in embed_models.items()
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


# ─── Роутер ───────────────────────────────────────────────────────────────────

router = Router()


# ─── Команды ──────────────────────────────────────────────────────────────────

@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    uid = message.from_user.id
    s = get_settings(uid)
    prov = PROVIDERS[s["provider"]]
    model_label = prov["models"].get(s["model"], {}).get("label", s["model"])
    await message.answer(
        "👋 Привет! Я бот с <b>долгой памятью</b> (RAG + ChromaDB).\n\n"
        f"Провайдер: <b>{prov['name']}</b> | Модель: <b>{model_label}</b>\n"
        f"Эмбеддинги: <b>{s.get('embed_model', prov['default_embed'])}</b>\n\n"
        "Загрузи PDF, TXT или DOCX — я проиндексирую его.\n"
        "Затем задай любой вопрос по документу.\n\n"
        "/config — настройки  |  /docs — мои документы  |  /clear — очистить базу",
    )


@router.message(Command("info"))
async def cmd_info(message: Message) -> None:
    uid = message.from_user.id
    s = get_settings(uid)
    prov = PROVIDERS[s["provider"]]
    model_label = prov["models"].get(s["model"], {}).get("label", s["model"])
    docs = user_docs.get(uid, [])
    await message.answer(
        f"⚙️ <b>Настройки</b>\n\n"
        f"Провайдер:     <b>{prov['name']}</b>\n"
        f"Модель:        <b>{model_label}</b>\n"
        f"Температура:   <b>{s['temperature']}</b>\n"
        f"Макс. токенов: <b>{s['max_tokens']}</b>\n"
        f"Эмбеддинги:    <b>{s.get('embed_model', prov['default_embed'])}</b>\n"
        f"Документов:    <b>{len(docs)}</b>",
    )


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
    uid = message.from_user.id
    collection = get_collection()
    existing = collection.get(where={"user_id": str(uid)})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
    user_docs.pop(uid, None)
    await message.answer("🗑 Все твои документы удалены из базы.")


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
    prov_name = PROVIDERS[provider_key]["name"]
    await call.message.edit_text(
        f"Провайдер: <b>{prov_name}</b>\nВыбери модель:",
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
        f"Введи температуру от {lo} до {hi} (например: <code>0.2</code>):",
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
    await _save_settings(call, state, provider_key, model_id, data.get("temperature", DEFAULT_TEMP), max_tokens)


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
    await _save_settings(message, state, provider_key, model_id, data.get("temperature", DEFAULT_TEMP), max_tokens)


async def _save_settings(event, state, provider_key, model_id, temperature, max_tokens) -> None:
    """После выбора токенов — предлагаем выбрать модель эмбеддингов."""
    await state.update_data(
        provider=provider_key, model=model_id,
        temperature=temperature, max_tokens=max_tokens,
    )
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
    """Пользователь выбрал модель эмбеддингов — финальное сохранение настроек."""
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
        f"✅ Сохранено!\n\n"
        f"Провайдер:     <b>{prov['name']}</b>\n"
        f"Модель:        <b>{model_label}</b>\n"
        f"Температура:   <b>{data['temperature']}</b>\n"
        f"Макс. токенов: <b>{data['max_tokens']}</b>\n"
        f"Эмбеддинги:    <b>{embed_label}</b>",
    )
    await call.answer()


# ─── Загрузка документа ───────────────────────────────────────────────────────

@router.message(F.document)
async def on_document(message: Message, bot: Bot) -> None:
    """
    Обрабатывает загрузку файла:
    1. Скачивает файл через Telegram API.
    2. Парсит текст (PDF/TXT/DOCX).
    3. Разбивает на чанки и сохраняет эмбеддинги в ChromaDB.
    """
    uid = message.from_user.id
    tg_doc = message.document

    # Используем провайдера из настроек пользователя для эмбеддингов
    s = get_settings(uid)
    provider = PROVIDERS[s["provider"]]
    embed_api_key = os.getenv(provider["api_key_env"])
    if not embed_api_key:
        await message.answer(f"⚠️ Не найден ключ {provider['api_key_env']} в .env")
        return

    embed_client = OpenAI(api_key=embed_api_key, base_url=provider["base_url"])

    # Скачиваем файл
    ensure_dirs()
    user_dir = Path(UPLOADS_DIR) / str(uid)
    user_dir.mkdir(parents=True, exist_ok=True)
    filename = tg_doc.file_name or f"file_{uuid.uuid4().hex}"
    save_path = user_dir / filename

    try:
        file_info = await bot.get_file(tg_doc.file_id)
        file_url = f"https://api.telegram.org/file/bot{bot.token}/{file_info.file_path}"
        async with aiohttp.ClientSession() as session:
            async with session.get(file_url) as resp:
                resp.raise_for_status()
                save_path.write_bytes(await resp.read())
    except Exception as e:
        logging.exception("Ошибка скачивания файла")
        await message.answer(f"❌ Не удалось скачать файл: {e}")
        return

    # Парсим и чанкуем документ
    await message.answer("⏳ Обрабатываю документ...")
    try:
        chunks = load_document(str(save_path))
    except Exception as e:
        await message.answer(f"❌ {e}")
        return

    if not chunks:
        await message.answer("⚠️ Не удалось извлечь текст из документа.")
        return

    # Сохраняем эмбеддинги в ChromaDB
    doc_id = uuid.uuid4().hex
    embed_model = s.get("embed_model", PROVIDERS[s["provider"]]["default_embed"])
    try:
        count = await asyncio.to_thread(embed_chunks, embed_client, uid, doc_id, chunks, embed_model)
    except Exception as e:
        logging.exception("Ошибка эмбеддинга/ChromaDB")
        await message.answer(f"❌ Ошибка при индексации: {e}")
        return

    # Запоминаем документ в памяти сессии
    user_docs.setdefault(uid, []).append({"doc_id": doc_id, "filename": filename, "chunks": count})

    await message.answer(
        f"✅ Документ <b>{filename}</b> проиндексирован.\n"
        f"Чанков сохранено: <b>{count}</b>\n\n"
        "Теперь задай вопрос по документу."
    )


# ─── Вопрос по документу ─────────────────────────────────────────────────────

@router.message(F.text)
async def on_question(message: Message) -> None:
    """
    RAG-пайплайн:
    1. Эмбеддинг вопроса.
    2. Поиск top-K релевантных чанков в ChromaDB.
    3. Передача контекста + вопроса в Chat Completions.
    4. Ответ пользователю.
    """
    uid = message.from_user.id
    question = message.text.strip()
    s = get_settings(uid)

    provider = PROVIDERS[s["provider"]]
    chat_api_key = os.getenv(provider["api_key_env"])

    if not chat_api_key:
        await message.answer(f"⚠️ Не найден ключ {provider['api_key_env']} в .env")
        return

    # Один клиент — и для эмбеддингов, и для чата (один провайдер)
    embed_client = OpenAI(api_key=chat_api_key, base_url=provider["base_url"])
    chat_client  = OpenAI(api_key=chat_api_key, base_url=provider["base_url"])

    await message.bot.send_chat_action(message.chat.id, "typing")

    # Поиск релевантных фрагментов
    embed_model = s.get("embed_model", PROVIDERS[s["provider"]]["default_embed"])
    try:
        context_chunks = await asyncio.to_thread(
            retrieve_context, embed_client, uid, question, embed_model, TOP_K
        )
    except Exception as e:
        logging.exception("Ошибка поиска в ChromaDB")
        await message.answer(f"❌ Ошибка поиска: {e}")
        return

    if not context_chunks:
        await message.answer(
            "📭 Контекст не найден. Загрузи документ командой /docs или отправь файл."
        )
        return

    # Генерация ответа
    try:
        reply = await answer_question(
            chat_client, context_chunks, question,
            s["model"], s["temperature"], s["max_tokens"],
        )
    except Exception as e:
        logging.exception("Ошибка Chat Completions")
        await message.answer(f"❌ Ошибка модели: {e}")
        return

    await message.answer(reply)


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

    logging.info("Бот (долгая память) запущен.")
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
