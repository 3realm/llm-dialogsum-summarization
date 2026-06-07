# LLM DialogSum Summarization

Практическая работа по суммаризации диалогов с использованием LLM.

В проекте сравниваются несколько подходов к задаче dialogue summarization на датасете DialogSum:

1. rule-based baseline: первое + последнее предложение диалога;
2. zero-shot LLM baseline через vLLM;
3. LoRA fine-tuning маленькой decoder-only модели;
4. дополнительная оценка качества через LLM-as-Judge.

---

## Dataset

Используется датасет DialogSum с Kaggle:

`marawanxmamdouh/dialogsum`

Основные колонки датасета:

* `dialogue` — исходный диалог;
* `summary` — эталонное краткое содержание;
* `topic` — тема диалога;
* `id` — технический идентификатор записи.

В рамках задачи основным входом модели является `dialogue`, а целевым ответом — `summary`.

---

## Project structure

```text
llm-dialogsum-summarization/
│
├── notebooks/
│   ├── 01_eda_and_baseline.ipynb
│   ├── 02_llm_vllm_inference.ipynb
│   ├── 03_lora_finetuning.ipynb
│   └── 04_llm_as_judge.ipynb
│
├── outputs/
│   ├── metrics/
│   └── predictions/
│
├── src/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Notebooks

### `01_eda_and_baseline.ipynb`

В этом ноутбуке выполнены:

* загрузка DialogSum;
* первичный EDA;
* анализ структуры данных;
* анализ длины диалогов и summary;
* построение rule-based baseline;
* расчёт BLEU и ROUGE.

Rule-based baseline строится по простому правилу:

```text
summary = первое предложение диалога + последнее предложение диалога
```

---

### `02_llm_vllm_inference.ipynb`

В этом ноутбуке выполнен zero-shot inference через vLLM.

Используемая модель:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Модель использовалась без дообучения. Ей передавался prompt с инструкцией:

```text
Summarize the following dialogue in one concise paragraph.
```

Такой режим называется zero-shot inference, потому что модель не обучалась на DialogSum и не получала few-shot примеры в prompt.

---

### `03_lora_finetuning.ipynb`

В этом ноутбуке выполнено LoRA fine-tuning той же модели.

Для обучения использовались:

* Unsloth;
* TRL / SFTTrainer;
* подмножество train set;
* LoRA adapters вместо полного обучения модели.

Во время обучения обновлялись только LoRA-адаптеры, а не все веса модели.

LoRA adapter был сохранён локально отдельно, но не добавлен в GitHub, так как это model artifact.

---

### `04_llm_as_judge.ipynb`

В этом ноутбуке выполнена дополнительная оценка качества через LLM-as-Judge.

LLM-судья сравнивал три варианта summary:

1. first + last sentence baseline;
2. Qwen zero-shot summary;
3. Qwen + LoRA summary.

Оценка выполнялась по критериям:

* factual correctness;
* coverage;
* conciseness;
* fluency;
* hallucination control.

---

## Metrics

Для автоматической оценки использовались:

* BLEU;
* ROUGE-1;
* ROUGE-2;
* ROUGE-L;
* ROUGE-Lsum.

Финальное сравнение на 100 test-примерах:

| Method                | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --------------------- | ---: | ------: | ------: | ------: |
| First + Last baseline | 4.35 |   0.188 |   0.020 |   0.153 |
| Qwen zero-shot        | 5.38 |   0.225 |   0.064 |   0.176 |
| Qwen + LoRA           | 6.61 |   0.260 |   0.072 |   0.196 |

---

## Results

По результатам экспериментов:

* rule-based baseline оказался самым слабым подходом, так как он не понимает смысл диалога и просто берёт фрагменты текста;
* Qwen zero-shot baseline улучшил качество по сравнению с rule-based baseline;
* LoRA fine-tuning дал дополнительный прирост по BLEU и ROUGE;
* LLM-as-Judge использовался как дополнительная смысловая оценка, так как BLEU и ROUGE оценивают в основном лексическое совпадение с reference summary.

---

## Technical notes

В ходе работы возникли и были решены несколько технических проблем окружения:

* конфликт CUDA-версий при установке vLLM;
* ошибка `libcudart.so.13`;
* проблема multiprocessing в vLLM: `fork` vs `spawn`;
* конфликт Pillow / PIL;
* ошибка bitsandbytes при 4-bit training: `Missing dependency: libnvJitLink.so.13`;
* ошибка сериализации checkpoint в TRL: `PicklingError: Can't pickle SFTConfig`.

Для LoRA fine-tuning было принято решение отключить 4-bit загрузку и использовать обычный `adamw_torch`, так как выбранная модель достаточно маленькая для обучения LoRA в Colab.

---

## Artifacts

В GitHub сохраняются:

* notebooks;
* метрики;
* prediction-файлы;
* результаты LLM-as-Judge.

Не сохраняются в GitHub:

* скачанные датасеты;
* checkpoints;
* LoRA adapter archive;
* API keys;
* Kaggle/Groq credentials.

LoRA adapter сохранён локально отдельно как model artifact.

---

## Conclusion

Практическая работа показывает полный базовый pipeline LLM-задачи:

```text
dataset
→ EDA
→ rule-based baseline
→ LLM zero-shot inference
→ LoRA fine-tuning
→ automatic metrics
→ LLM-as-Judge evaluation
```

Даже короткое LoRA fine-tuning на ограниченном числе шагов позволило улучшить качество summary по сравнению с zero-shot baseline. При этом автоматические метрики были дополнены LLM-as-Judge, так как для summarization важно оценивать не только совпадение слов, но и смысловое качество ответа.
