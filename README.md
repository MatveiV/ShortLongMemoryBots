# ShortLongMemory Bot — Документация проекта

Проект реализует три Telegram-бота с разными типами памяти на базе aiogram 3.x и OpenAI-совместимых API.

---

## Структура проекта

```
ShortLongMemory_bot/
├── bot_short_memory.py       # Бот с короткой памятью
├── bot_long_memory.py        # Бот с долгой памятью (RAG)
├── bot_shortlong_memory.py   # Универсальный бот (выбор режима)
├── ai_direct.py              # CLI-интерфейс для работы с AI
├── openai_client.py          # Унифицированный OpenAI-клиент
├── .env                      # API-ключи и токены
├── requirements.txt          # Зависимости
├── memory/                   # ChromaDB persistent storage
└── uploads/                  # Загруженные пользователями файлы
```

---

## Быстрый старт

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
pip install chromadb pypdf python-docx aiohttp --prefer-binary

python bot_shortlong_memory.py   # универсальный бот
# или
python bot_short_memory.py
python bot_long_memory.py
```

---

## Конфигурация (.env)

```env
BOT_TOKEN=your-telegram-bot-token

ZAI_API_KEY=your-zai-key
PROXY_API_KEY=your-proxyapi-key
GEN_API_KEY=your-genapi-key

EMBED_MODEL=text-embedding-3-small   # опционально
```

---

## Провайдеры и модели

Все три бота используют одинаковую структуру провайдеров:

| Провайдер       | Переменная      | URL                                  |
|-----------------|-----------------|--------------------------------------|
| Z.AI            | ZAI_API_KEY     | https://api.z.ai/api/paas/v4/        |
| ProxyAPI        | PROXY_API_KEY   | https://api.proxyapi.ru/openai/v1    |
| GenAPI          | GEN_API_KEY     | https://proxy.gen-api.ru/v1          |

### Чат-модели Z.AI
| Модель        | Бесплатно | Макс. токенов | Температура |
|---------------|-----------|---------------|-------------|
| GLM-4.7-Flash | ✅        | 4 096         | 0.0–1.0     |
| GLM-4.5-Flash | ✅        | 4 096         | 0.0–1.0     |
| GLM-4.7       | ❌        | 8 192         | 0.0–1.0     |
| GLM-4.5       | ❌        | 8 192         | 0.0–1.0     |

### Чат-модели ProxyAPI
| Модель       | Макс. токенов | Температура |
|--------------|---------------|-------------|
| GPT-4.1 Nano | 32 768        | 0.0–2.0     |
| GPT-4.1 Mini | 32 768        | 0.0–2.0     |
| GPT-4.1      | 32 768        | 0.0–2.0     |
| GPT-4o Mini  | 16 384        | 0.0–2.0     |
| GPT-4o       | 16 384        | 0.0–2.0     |

### Чат-модели GenAPI
| Модель            | Макс. токенов | Температура |
|-------------------|---------------|-------------|
| GPT-4.1 Mini      | 32 768        | 0.0–2.0     |
| GPT-4.1           | 32 768        | 0.0–2.0     |
| GPT-4o            | 16 384        | 0.0–2.0     |
| Claude Sonnet 4.5 | 8 192         | 0.0–1.0     |
| Gemini 2.5 Flash  | 8 192         | 0.0–2.0     |
| DeepSeek Chat     | 8 192         | 0.0–2.0     |
| DeepSeek R1       | 16 000        | 0.0–2.0     |

### Модели эмбеддингов (для долгой памяти)
| Провайдер | Модели                                                          |
|-----------|-----------------------------------------------------------------|
| Z.AI      | embedding-3, embedding-2                                        |
| ProxyAPI  | text-embedding-3-small, text-embedding-3-large, ada-002         |
| GenAPI    | text-embedding-3-small, text-embedding-3-large, ada-002         |

---

## Описание ботов

### bot_short_memory.py — Короткая память

Хранит последние 10 сообщений диалога в оперативной памяти.

Команды: `/start` `/config` `/new` `/info`

### bot_long_memory.py — Долгая память (RAG)

Индексирует загруженные документы через эмбеддинги в ChromaDB. Отвечает строго по документу.

Команды: `/start` `/config` `/docs` `/clear` `/info`

### bot_shortlong_memory.py — Универсальный бот

После `/start` предлагает выбрать режим:
- 💬 Короткая память
- 📚 Долгая память
- 🧠 Короткая + Долгая память

Команды: `/start` `/config` `/new` `/docs` `/clear` `/info`

---

## Архитектура памяти

### Короткая память

```
RAM: Dict[user_id, deque(maxlen=10)]

Каждое сообщение: {"role": "user"|"assistant", "content": "..."}

Жизненный цикл:
  - Создаётся при первом сообщении
  - Очищается командой /new
  - Сбрасывается при перезапуске бота
```

### Долгая память

```
ChromaDB (./memory/) — persistent vector store

Каждый чанк:
  - id:       "user_id:doc_id:chunk_index"
  - document: текст чанка (~500 символов)
  - embedding: вектор float[] (размерность зависит от модели)
  - metadata: {user_id, doc_id, chunk_index}

Жизненный цикл:
  - Создаётся при загрузке документа
  - Сохраняется между перезапусками
  - Удаляется командой /clear
```

### Комбинированная память

```
Короткая: deque(maxlen=10) в RAM
Долгая:   ChromaDB (persistent)

При запросе:
  1. Поиск в ChromaDB → top-5 чанков
  2. История из deque
  3. Объединение в один prompt → модель
```

---

## Диаграммы

> Mermaid-версии рендерятся прямо на GitHub. PlantUML-версии можно использовать локально или через [plantuml.com](https://www.plantuml.com/plantuml/uml/).

### Диаграмма 1: Архитектура системы

```mermaid
graph TD
    User([Пользователь]) -->|сообщение / файл| TG[Telegram Bot API]
    TG -->|polling| AIOGRAM[aiogram 3.x\nRouter / Dispatcher]

    AIOGRAM -->|/config| FSM[FSM ConfigStates]
    AIOGRAM --> SETTINGS[User Settings\nDict uid→dict]
    AIOGRAM --> DEQUE[deque maxlen=10\nКороткая память]
    AIOGRAM --> PARSER[Document Parser\nPDF / TXT / DOCX]
    PARSER --> EMBED[Embeddings API]
    EMBED --> CHROMA[(ChromaDB\n./memory/)]
    AIOGRAM -->|query top-K| CHROMA
    AIOGRAM --> LLM[Chat Completions]

    LLM --> ZAI[Z.AI\napi.z.ai]
    LLM --> PROXY[ProxyAPI\napi.proxyapi.ru]
    LLM --> GEN[GenAPI\nproxy.gen-api.ru]
    EMBED --> ZAI
    EMBED --> PROXY
    EMBED --> GEN
```

<details>
<summary>PlantUML</summary>

```plantuml
@startuml architecture
!theme plain
skinparam backgroundColor #FAFAFA
skinparam componentStyle rectangle

package "Telegram" {
  actor User
  component "Telegram Bot API" as TG
}

package "Bot Application" {
  component "aiogram 3.x\nRouter / Dispatcher" as AIOGRAM
  component "FSM\n(ConfigStates)" as FSM
  component "User Settings\nDict[uid, dict]" as SETTINGS

  package "Короткая память" {
    component "deque(maxlen=10)\nDict[uid, Deque]" as DEQUE
  }

  package "Долгая память" {
    component "ChromaDB\nPersistentClient" as CHROMA
    component "Embeddings API" as EMBED
    component "Document Parser\nPDF/TXT/DOCX" as PARSER
  }

  component "OpenAI-compatible\nChat Completions" as LLM
}

package "AI Providers" {
  component "Z.AI\napi.z.ai" as ZAI
  component "ProxyAPI\napi.proxyapi.ru" as PROXY
  component "GenAPI\nproxy.gen-api.ru" as GEN
}

User --> TG : сообщение / файл
TG --> AIOGRAM : webhook / polling
AIOGRAM --> FSM : /config
AIOGRAM --> SETTINGS : get_settings()
AIOGRAM --> DEQUE : короткая память
AIOGRAM --> PARSER : загрузка файла
PARSER --> EMBED : текст → эмбеддинги
EMBED --> CHROMA : upsert()
AIOGRAM --> CHROMA : query() top-K
AIOGRAM --> LLM : chat.completions.create()
LLM --> ZAI
LLM --> PROXY
LLM --> GEN
EMBED --> ZAI
EMBED --> PROXY
EMBED --> GEN
@enduml
```

</details>

---

### Диаграмма 2: Поток обработки сообщения — Короткая память

```mermaid
flowchart TD
    A([Пользователь отправляет текст]) --> B[get_settings uid]
    B --> C[short_memory uid .append role:user]
    C --> D["Payload: [system] + list(deque) + [user]"]
    D --> E[chat.completions.create]
    E --> F{Ошибка API?}
    F -->|да| G[Удалить из deque\nОтправить ошибку]
    F -->|нет| H[Получить ответ модели]
    H --> I[Отправить ответ пользователю]
    I --> J[short_memory uid .append role:assistant]
```

<details>
<summary>PlantUML</summary>

```plantuml
@startuml short_memory_flow
!theme plain
skinparam backgroundColor #FAFAFA

title bot_short_memory.py — Обработка сообщения

start

:Пользователь отправляет текст;
:Получить настройки пользователя\nget_settings(uid);
:Добавить сообщение в deque\nshort_memory[uid].append({role:user});

:Сформировать payload:\n[system_msg] + list(deque) + [user_msg];

:Отправить запрос\nclient.chat.completions.create();

if (Ошибка API?) then (да)
  :Удалить последнее сообщение из deque;
  :Отправить сообщение об ошибке;
  stop
else (нет)
  :Получить ответ модели;
  :Отправить ответ пользователю;
  :Добавить ответ в deque\nshort_memory[uid].append({role:assistant});
endif

stop
@enduml
```

</details>

---

### Диаграмма 3: Поток обработки документа и вопроса — Долгая память

```mermaid
flowchart TD
    subgraph DOC[Загрузка документа]
        D1([Пользователь загружает файл]) --> D2[Скачать через Telegram API\naiohttp]
        D2 --> D3[load_document → chunk_text\nsize=500 overlap=50]
        D3 --> D4[embed_chunks\nembeddings.create model input=chunks]
        D4 --> D5[(ChromaDB.upsert\nid embedding metadata document)]
        D5 --> D6[Сообщить количество чанков]
    end

    subgraph QA[Ответ на вопрос]
        Q1([Пользователь задаёт вопрос]) --> Q2[retrieve_context\nembeddings.create input=question]
        Q2 --> Q3[(ChromaDB.query\ntop_k=5 where=user_id)]
        Q3 --> Q4{Чанки найдены?}
        Q4 -->|нет| Q5[Сообщить: загрузи документ]
        Q4 -->|да| Q6["Payload: [system] + [контекст + вопрос]"]
        Q6 --> Q7[chat.completions.create]
        Q7 --> Q8[Отправить ответ]
    end
```

<details>
<summary>PlantUML</summary>

```plantuml
@startuml long_memory_flow
!theme plain
skinparam backgroundColor #FAFAFA

title bot_long_memory.py — RAG Pipeline

partition "Загрузка документа" {
  start
  :Пользователь загружает файл\n(PDF / TXT / DOCX);
  :Скачать файл через Telegram API\n(aiohttp);
  :load_document() → chunk_text()\nsize=500, overlap=50;
  :embed_chunks()\nembeddings.create(model=embed_model, input=chunks);
  :ChromaDB.upsert()\nid, embedding, metadata, document;
  :Сообщить пользователю о количестве чанков;
  stop
}

partition "Ответ на вопрос" {
  start
  :Пользователь задаёт вопрос;
  :retrieve_context()\nembeddings.create(input=[question]);
  :ChromaDB.query()\ntop_k=5, where={user_id};

  if (Чанки найдены?) then (нет)
    :Сообщить: "Загрузи документ";
    stop
  else (да)
    :Сформировать prompt:\nsystem + [контекст] + вопрос;
    :chat.completions.create();
    :Отправить ответ пользователю;
    stop
  endif
}
@enduml
```

</details>

---

### Диаграмма 4: Комбинированный пайплайн — Короткая + Долгая память

```mermaid
flowchart TD
    A([Пользователь отправляет текст]) --> B{Режим?}

    B -->|MODE_SHORT| S1[Добавить в deque]
    S1 --> S2["Payload: [system] + history + [user]"]
    S2 --> S3[chat.completions.create]
    S3 --> S4[Добавить ответ в deque]

    B -->|MODE_LONG| L1[(retrieve_context → ChromaDB)]
    L1 --> L2{Контекст найден?}
    L2 -->|нет| L3[Сообщить об отсутствии документов]
    L2 -->|да| L4["Payload: [system] + [контекст + вопрос]"]
    L4 --> L5[chat.completions.create]

    B -->|MODE_COMBINED| C1[Добавить в deque]
    C1 --> C2[(retrieve_context → ChromaDB)]
    C2 --> C3{Контекст найден?}
    C3 -->|да| C4["user_content = контекст + вопрос"]
    C3 -->|нет| C5[user_content = вопрос]
    C4 --> C6["Payload: [system] + history + [user_content]"]
    C5 --> C6
    C6 --> C7[chat.completions.create]
    C7 --> C8[Добавить ответ в deque]

    S4 --> END([Отправить ответ])
    L5 --> END
    C8 --> END
```

<details>
<summary>PlantUML</summary>

```plantuml
@startuml combined_flow
!theme plain
skinparam backgroundColor #FAFAFA

title bot_shortlong_memory.py — answer_with_memory()

start

:Пользователь отправляет текст;
:Определить режим\nuser_settings[uid]["mode"];

switch (Режим?)
case (MODE_SHORT)
  :Добавить в deque;
  :Payload: [system] + history + [user];
  :chat.completions.create();
  :Добавить ответ в deque;

case (MODE_LONG)
  :retrieve_context() → ChromaDB;
  if (Контекст найден?) then (нет)
    :Сообщить об отсутствии документов;
    stop
  else (да)
    :Payload: [system] + [контекст + вопрос];
    :chat.completions.create();
  endif

case (MODE_COMBINED)
  :Добавить в deque;
  :retrieve_context() → ChromaDB;
  :history = list(deque);
  note right
    Если контекст найден:
    user_content = контекст + вопрос
    Если нет:
    user_content = вопрос
  end note
  :Payload: [system] + history + [user_content];
  :chat.completions.create();
  :Добавить ответ в deque;
endswitch

:Отправить ответ пользователю;
stop
@enduml
```

</details>

---

### Диаграмма 5: FSM — Настройка /config

```mermaid
stateDiagram-v2
    [*] --> ProviderSelect : /config
    ProviderSelect --> ModelSelect : prov callback
    ModelSelect --> ProviderSelect : Назад
    ModelSelect --> Temp : model callback
    Temp --> Tokens : валидный float
    Tokens --> Embed : tokens callback / ручной ввод
    Embed --> Done : embed callback
    Done --> [*]

    note right of ProviderSelect : inline-кнопки\nZ.AI / ProxyAPI / GenAPI
    note right of ModelSelect : список моделей провайдера
    note right of Temp : float в диапазоне модели
    note right of Tokens : кнопки или ручной ввод
    note right of Embed : только для long и combined\nbot_short_memory - этот шаг пропускается
    note right of Done : user_settings обновлён
```

<details>
<summary>PlantUML</summary>

```plantuml
@startuml fsm_config
!theme plain
skinparam backgroundColor #FAFAFA

title FSM — /config (bot_shortlong_memory.py)

[*] --> ProviderSelect : /config

state "Выбор провайдера" as ProviderSelect
state "Выбор модели" as ModelSelect
state "waiting_temperature" as Temp
state "waiting_max_tokens" as Tokens
state "waiting_embed_model" as Embed
state "Сохранено" as Done

note right of ProviderSelect : inline-кнопки\nZ.AI / ProxyAPI / GenAPI
note right of ModelSelect : inline-кнопки\nсписок моделей провайдера
note right of Temp : текстовый ввод\nfloat в диапазоне модели
note right of Tokens : inline-кнопки или\nтекстовый ввод
note right of Embed : inline-кнопки\nмодели эмбеддингов\n(только long/combined)
note right of Done : user_settings[uid] обновлён

ProviderSelect --> ModelSelect : prov:* callback
ModelSelect --> Temp : model:* callback
ModelSelect --> ProviderSelect : Назад
Temp --> Tokens : валидный float
Tokens --> Embed : tokens:* callback
Tokens --> Embed : ручной ввод
Embed --> Done : embed:* callback
Done --> [*]

note bottom of Tokens
  bot_short_memory.py:
  Tokens --> Done (без шага Embed)
end note
@enduml
```

</details>

---

### Диаграмма 6: Структура памяти в ChromaDB

```mermaid
classDiagram
    class Chunk {
        +id: uid:doc_id:chunk_index
        +document: str ~500 символов
        +embedding: List~float~
        +metadata_user_id: str
        +metadata_doc_id: str
        +metadata_chunk_index: int
    }

    class UserDocs {
        +user_id: int
        +doc_id: str
        +filename: str
        +chunks: int
    }

    class ShortMemory {
        +user_id: int
        +role: user | assistant
        +content: str
        +maxlen: 10
    }

    Chunk "1..*" --> UserDocs : doc_id
```

<details>
<summary>PlantUML</summary>

```plantuml
@startuml chroma_structure
!theme plain
skinparam backgroundColor #FAFAFA

title Структура данных в ChromaDB

package "ChromaDB (./memory/)" {
  package "Коллекция: long_memory\n(hnsw:space=cosine)" {

    class Chunk {
      + id: "uid:doc_id:chunk_index"
      + document: str  ~~500 символов~~
      + embedding: List[float]
      + metadata.user_id: str
      + metadata.doc_id: str
      + metadata.chunk_index: int
    }

    note bottom of Chunk
      Поиск: cosine similarity
      Фильтр: where={user_id: str(uid)}
      top_k = 5 по умолчанию
    end note
  }
}

package "RAM (сессия)" {
  class UserDocs {
    + user_id: int
    + docs: List[dict]
    --
    doc_id: str
    filename: str
    chunks: int
  }

  class ShortMemory {
    + user_id: int
    + history: deque(maxlen=10)
    --
    role: "user" | "assistant"
    content: str
  }
}

Chunk "1..*" -- UserDocs : doc_id
@enduml
```

</details>

---

### Диаграмма 7: Выбор режима после /start

```mermaid
stateDiagram-v2
    [*] --> ModeSelect : /start

    state "Выбор режима" as ModeSelect
    state "Короткая память\ndeque maxlen=10\n/new /config /info" as Short
    state "Долгая память\nChromaDB RAG\n/docs /clear /config /info" as Long
    state "Короткая + Долгая\ndeque + ChromaDB\n/new /docs /clear /config /info" as Combined

    ModeSelect --> Short : mode:short
    ModeSelect --> Long : mode:long
    ModeSelect --> Combined : mode:combined

    Short --> ModeSelect : /start
    Long --> ModeSelect : /start
    Combined --> ModeSelect : /start
```

<details>
<summary>PlantUML</summary>

```plantuml
@startuml mode_selection
!theme plain
skinparam backgroundColor #FAFAFA

title bot_shortlong_memory.py — Выбор режима

[*] --> Start : /start

state "Выбор режима" as Start
state "Короткая память" as Short
state "Долгая память" as Long
state "Короткая + Долгая" as Combined

note right of Start : Показать inline-кнопки\nвыбора режима
note right of Short : deque(maxlen=10)\nтолько история диалога\nКоманды: /new /config /info
note right of Long : ChromaDB RAG\nтолько по документам\nКоманды: /docs /clear /config /info
note right of Combined : deque + ChromaDB\nистория + документы\nКоманды: /new /docs /clear /config /info

Start --> Short    : mode:short
Start --> Long     : mode:long
Start --> Combined : mode:combined

Short    --> Start : /start
Long     --> Start : /start
Combined --> Start : /start
@enduml
```

</details>

---

## Как работает каждый тип памяти

### Короткая память

Реализована через `collections.deque(maxlen=10)` — кольцевой буфер в оперативной памяти.

Библиотеки: `collections` (stdlib)

Хранение: только в RAM, не персистентна.

Логика запроса:
```
[system_prompt]
[user msg N-9]
[assistant msg N-9]
...
[user msg N]   ← текущий вопрос
```
Модель видит контекст последних 10 реплик. При переполнении старые сообщения автоматически вытесняются.

Очистка: `/new` — `deque.clear()`

---

### Долгая память

Реализована через ChromaDB — локальную векторную базу данных с HNSW-индексом.

Библиотеки: `chromadb`, `openai` (embeddings), `pypdf`, `python-docx`, `aiohttp`

Хранение: персистентно в папке `./memory/`, сохраняется между перезапусками.

Логика индексации:
```
Файл → load_document() → chunk_text(size=500, overlap=50)
     → embeddings.create(model, input=chunks)
     → ChromaDB.upsert(ids, embeddings, metadatas, documents)
```

Логика поиска:
```
Вопрос → embeddings.create(model, input=[question])
       → ChromaDB.query(query_embeddings, n_results=5, where={user_id})
       → top-5 чанков по cosine similarity
```

Логика запроса к модели:
```
[system: "Отвечай только по контексту"]
[user: "Контекст:\n{chunks}\n\nВопрос: {question}"]
```

Очистка: `/clear` — `ChromaDB.delete(ids)` для всех чанков пользователя.

---

### Комбинированная память

Объединяет оба подхода в функции `answer_with_memory()`.

Логика запроса:
```
[system: "Используй документ и историю"]
[user msg N-9]          ← короткая память
[assistant msg N-9]
...
[user: "Контекст:\n{chunks}\n\nВопрос: {question}"]  ← долгая память + текущий вопрос
```

Приоритет: если в ChromaDB найдены релевантные фрагменты — они включаются в запрос. Если нет — запрос идёт только с историей диалога.

---

## Зависимости

```
aiogram>=3.7.0
openai>=1.0.0
python-dotenv
chromadb
pypdf
python-docx
aiohttp
```
