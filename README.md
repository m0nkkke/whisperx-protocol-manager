# 🎙️ WhisperX Protocol Manager

Небольшое веб-приложение на FastAPI для загрузки и транскрибации аудиофайлов с поддержкой диаризации спикеров и генерации кратких протоколов содержания.

## 🚀 Возможности

1. 📤 Загрузка аудиофайлов форматов: `ogg`, `wav`, `mp3`, и др.
2. 📝 Транскрибация аудио с возможностью скачать текст.
3. 🧑‍🤝‍🧑 Поддержка **диаризации спикеров** (определение "кто и когда говорил") с возможностью указания количества говорящих.
4. 🧠 Генерация **протокола/конспекта** по результатам транскрипции.
5. ⚙️ Настройка параметров транскрибации (модель, качество и др.).
6. 💬 Возможность задавать **промпты** (подсказки) для улучшения транскрибации и протоколирования.

## 📸 Интерфейс приложения

[![Главный интерфейс приложения](https://i.ibb.co/Z1R5nnvF/wxpm.png)](https://ibb.co/vvCp77fT)

## 📂 Структура проекта

```
├── app.py                  # основной FastAPI сервер
├── index.html              # веб-интерфейс
├── styles.css              # стили
├── requirements.txt        # зависимости проекта
├── .env                    # переменные окружения
├── prompts/                # пользовательские промпты (текстовые файлы .txt)
└── reference_voices/       # голосовые образцы для диаризации
```

## 🛠️ Технологии

- [FastAPI](https://fastapi.tiangolo.com/)
- [WhisperX](https://github.com/m-bain/whisperx)
- [PyAnnote-Audio](https://github.com/pyannote/pyannote-audio)
- [Hugging Face Transformers](https://huggingface.co/)
- [Gemini API (Google)](https://ai.google.dev/)

## 🧠 Диаризация и банк голосов

Для диаризации используется **голосовой банк (reference voices)**.  
Чтобы добавить свой голос, поместите **отрывок речи длиной от 15 до 30 секунд** в папку `reference_voices/`.  
Имя файла будет использоваться как метка (например, `ivan.wav` → `ivan:` в тексте).

> Если не указаны кастомные голоса, будет использована автоматическая кластеризация по количеству спикеров.

## 💬 Промпты

Для улучшения качества распознавания и конспектирования можно задать **промпты**.  
Разместите текстовые подсказки в виде `.txt` файлов в папке `prompts/`.  
Например:
- `context.txt` — описание ситуации (лекция, совещание и т.д.)
- `names.txt` — список возможных участников разговора
- `vocabulary.txt` — терминология по теме

## 📦 Установка

```bash
git clone https://github.com/yourusername/audio-transcriber-api.git
cd audio-transcriber-api
python -m venv venv
source venv/bin/activate  # или .\venv\Scripts\activate для Windows
pip install -r requirements.txt
```

Создайте .env файл в корне проекта:
```
# Токен Hugging Face для pyannote
HUGGINGFACE_TOKEN=ваш_токен

# API ключ Gemini
GEMINI_API_KEY=ваш_ключ

# Прокси (если необходимо)
HTTP_PROXY=
HTTPS_PROXY=

# Параметры WhisperX
MODEL_NAME=large-v3
COMPUTE_TYPE=int8

# Параметры диаризации (рекомендуемые)
MIN_SPEAKERS=2
MAX_SPEAKERS=2
SEGMENTATION_THRESHOLD=0.60
SEGMENTATION_MIN_DURATION_OFF=0.20
CLUSTERING_THRESHOLD=0.50
CLUSTERING_MIN_CLUSTER_SIZE=8
```
## 🔐 Требования к авторизации

### Hugging Face Access
Для работы с диаризацией через PyAnnote необходимо:
1. Получить токен на [Hugging Face](https://huggingface.co/settings/tokens)
2. **Обязательно подтвердить доступ к следующим моделям:**
   - [pyannote/segmentation](https://huggingface.co/pyannote/segmentation)
   - [pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)
3. Для каждой модели нужно:
   - Перейти по ссылке
   - Нажать "Agree to access"
   - Авторизоваться при необходимости

### Gemini API
Для генерации протоколов требуется:
1. Получить API ключ в [Google AI Studio](https://ai.google.dev/)
2. Добавить его в `.env` файл как `GEMINI_API_KEY`

▶️ Запуск
```
uvicorn app:app --reload
```

Откройте в браузере: http://localhost:8000

## ⚙️ Настраиваемые параметры

| Переменная                       | Описание                                                                 |
|----------------------------------|--------------------------------------------------------------------------|
| `MODEL_NAME`                     | Модель WhisperX (`base`, `medium`, `large-v3`)                           |
| `COMPUTE_TYPE`                   | Тип вычислений (`int8`, `float16`, `float32`)                            |
| `MIN_SPEAKERS`, `MAX_SPEAKERS`  | Количество предполагаемых спикеров (рекомендуется одинаковое значение)  |
| `SEGMENTATION_THRESHOLD`        | Порог вероятности для сегментации речи (оптимум `0.60`)                 |
| `SEGMENTATION_MIN_DURATION_OFF` | Мин. продолжительность тишины между сегментами (в секундах)             |
| `CLUSTERING_THRESHOLD`          | Порог объединения сегментов одного спикера                              |
| `CLUSTERING_MIN_CLUSTER_SIZE`   | Мин. количество сегментов в одном кластере                               |

## 📄 Примеры использования

- **Загрузка файла:**  
  Загрузите аудиофайл через веб-интерфейс.

- **Диаризация:**  
  Установите флаг "Диаризация", выберите количество спикеров или используйте кастомные голоса (папка `reference_voices/`).

- **Промпты:**  
  Добавьте текстовые подсказки в папку `prompts/` перед запуском транскрипции.
