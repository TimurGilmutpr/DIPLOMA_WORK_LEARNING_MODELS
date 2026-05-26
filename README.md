# DIPLOMA_WORK_LEARNING_MODELS

Репозиторий финальных экспериментов по обучению, объединению и production-проверке моделей для проекта **«AI-генетический консультант»** — системы русскоязычной интерпретации кариотипов в нотации ISCN.

Основной фокус репозитория — LoRA-адаптация LLaMA-3.1-8B-Instruct, сравнение с full fine-tuning и baseline, проверка инференса через vLLM и подготовка модели к сервисному использованию.

> ⚠️ Проект носит исследовательский характер. Сгенерированные интерпретации не являются медицинским заключением и должны проверяться специалистом.

---

## Роль репозитория в общей системе

Общая архитектура дипломного проекта включает пять логических блоков:

1. выбор базовой LLM;
2. подготовка специализированного ISCN Q&A-корпуса;
3. сравнение baseline, RAG, full fine-tuning и LoRA;
4. production-инференс через vLLM;
5. веб-сервис FastAPI для клинического MVP.

Данный репозиторий соответствует этапам **обучения, объединения LoRA-адаптера, тестирования vLLM и анализа финальных ответов модели**.

---

## Что находится в репозитории

```text
DIPLOMA_WORK_LEARNING_MODELS/
├── MERGE_LORA/          # объединение LoRA-адаптера с базовой моделью
├── TESTS_VLLM/          # тесты инференса через vLLM
├── TESTS_VLLM_RAG/      # тесты связки vLLM + RAG
├── TEST_ANSWERS/        # сохранённые ответы модели на тестовых примерах
├── ПРИМЕРЫ/             # примеры входов/выходов
└── README.md
```

---

## Цель

Разработать и экспериментально проверить модель, которая по ISCN-записи формирует структурированную русскоязычную интерпретацию для сценария ВРТ:

```text
46,XX,t(7;16)(p12.1;q23.1)
```

Пример ожидаемой логики ответа:

```text
Кариотип соответствует женскому хромосомному набору с реципрокной транслокацией между хромосомами 7 и 16. Точки разрыва расположены в регионах 7p12.1 и 16q23.1. Такая перестройка может иметь значение для репродуктивного прогноза и требует консультации врача-генетика.
```

---

## Базовая модель

В качестве базовой модели используется:

```text
meta-llama/Meta-Llama-3.1-8B-Instruct
```

Причины выбора:

- хорошая мультиязычная устойчивость;
- пригодность для русскоязычного сценария;
- совместимость с Hugging Face Transformers, PEFT и vLLM;
- достаточный баланс качества и вычислительной стоимости для MVP;
- возможность parameter-efficient fine-tuning через LoRA.

---

## Конфигурация LoRA-адаптации

Параметры из дипломного исследования:

| Параметр | Значение |
|---|---|
| Базовая модель | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| Библиотека | PEFT / Hugging Face |
| `r` | 16 |
| `lora_alpha` | 32 |
| `target_modules` | `q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `embed_tokens` |
| `lora_dropout` | 0.05 |
| optimizer | AdamW |
| learning rate | `2e-5` |
| epochs | 10 |
| batch size | 4 |
| gradient accumulation | 4 |
| scheduler | cosine |
| precision | bf16 |
| split | train 80% / validation 10% / test 10% |

---

## Результаты

Итоговое сравнение конфигураций:

| Конфигурация | Exact Match, % | BLEU, % | BERTScore F1, % | Latency, с | Integrated Score |
|---|---:|---:|---:|---:|---:|
| Baseline / Clear | 0.00 | 0.26 | 62.99 | 5.22 | — |
| Full fine-tuning | 0.00 | 19.30 | 81.80 | 16.04 | — |
| LoRA | 44.43 | 70.65 | 92.79 | 6.99 | 0.4834 |

Ключевой вывод: **LoRA показала лучший баланс качества и производительности**. Full fine-tuning улучшил качество относительно baseline, но оказался менее устойчивым на малом корпусе и значительно медленнее.

---

## Установка

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch transformers datasets accelerate peft bitsandbytes
pip install vllm pandas numpy tqdm scikit-learn jupyter
```

Для GPU-инференса через vLLM требуется CUDA-совместимая видеокарта и корректно установленный PyTorch/vLLM.

---

## Быстрый сценарий работы

### 1. Подготовить базовую модель и LoRA-адаптер

Убедитесь, что доступны:

```text
base_model/
lora_adapter/
```

или пути к моделям на Hugging Face / локальном диске.

### 2. Объединить LoRA с базовой моделью

Типовой код:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
adapter_path = "./lora_adapter"
output_path = "./LORA_THE_BEST"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()

model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
```

### 3. Запустить vLLM-сервер

```bash
vllm serve ./LORA_THE_BEST \
  --host 0.0.0.0 \
  --port 8001 \
  --served-model-name LORA_THE_BEST \
  --max-model-len 8192 \
  --max-num-seqs 4
```

### 4. Отправить тестовый запрос

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8001/v1", api_key="local-key")

response = client.chat.completions.create(
    model="LORA_THE_BEST",
    messages=[
        {"role": "system", "content": "Ты эксперт-цитогенетик. Отвечай на русском языке."},
        {"role": "user", "content": "Интерпретируй ISCN: 46,XX,t(7;16)(p12.1;q23.1)"}
    ],
    max_tokens=1600,
    temperature=0.7,
)

print(response.choices[0].message.content)
```

---

## Формат промпта

Рекомендуемый системный промпт:

```text
Ты эксперт-цитогенетик. Твоя задача — интерпретировать кариотипы в нотации ISCN на русском языке. Ответ должен быть структурированным, клинически нейтральным и не должен заменять заключение врача.
```

Рекомендуемые разделы ответа:

1. Расшифровка ISCN.
2. Тип аномалии.
3. Вовлечённые хромосомы и регионы.
4. Клиническое значение.
5. Репродуктивный контекст.
6. Рекомендация консультации специалиста.
7. Дисклеймер.

---

## vLLM vs Transformers pipeline

В дипломном исследовании vLLM выбран как предпочтительный production-инструмент, потому что:

- уменьшает latency при практическом сценарии;
- поддерживает continuous batching;
- лучше масштабируется при конкурентных запросах;
- совместим с OpenAI-compatible API;
- подходит для FastAPI-сервиса.

Transformers pipeline рекомендуется оставить для отладки и offline-экспериментов.

