# Исследовательский отчёт: mini-pretrain Qwen3 на  Wikipedia

## Данные и токенизация

- Датасет: `wikimedia/wikipedia`, конфигурация `20231101.ru`, split `train`.
- Первые 5000 примеров используются только для validation.
- Токенизатор: `ai-forever/rugpt3small_based_on_gpt2`.
- Длина каждой последовательности: 512 токенов.
- Padding находится справа. 

## Конфигурации экспериментов
Архитектура,данные, validation split, длина 512 и временной бюджет оставались неизменными.

| Profile | Batch | Accumulation | Effective batch | LR | Scheduler | Optimizer | Compile | Dtype |
|---|---:|---:|---:|---:|---|---|---|---|
| `baseline` | 4 | 1 | 4 | 5e-5 | linear | AdamW | нет | BF16 |
| `batch-accum` | 8 | 2 | 16 | 5e-5 | linear | AdamW | нет | BF16 |
| `large-batch` | 16 | 1 | 16 | 5e-5 | linear | AdamW | нет | BF16 |
| `compile-fused` | 16 | 1 | 16 | 5e-5 | linear | fused AdamW | да | BF16 + TF32 |
| `fp32-control` | 4 | 1 | 4 | 5e-5 | linear | AdamW | нет | FP32 |
| `optimized` | 16 | 1 | 16 | 3e-4 | cosine | fused AdamW | да | BF16 + TF32 |

## Результаты
 Perplexity вычислялась как
`exp(eval_loss)`. Чем меньше обе метрики, тем лучше модель предсказывает
следующий токен.

| Profile | Global step | Eval loss | Perplexity | Отчёт и график loss | Aviflow run |
|---|---:|---:|---:|---|---|
| `baseline` | 6421 | 6.4439 | 628.87 | [HTML](results/baseline.html) | [run](https://aviflow.k.avito.ru/_/pipeline/?ns=students#/runs/details/865b5eff-5e2e-4539-a760-9609ac369bde) |
| `batch-accum` | 2046 | 6.3053 | 547.46 | [HTML](results/batch-accum.html) | [run](https://aviflow.k.avito.ru/_/pipeline/?ns=students#/runs/details/9f0df634-ef27-4fba-873d-7ff0380f6554) |
| `large-batch` | 2148 | 6.2739 | 530.53 | [HTML](results/large-batch.html) | [run](https://aviflow.k.avito.ru/_/pipeline/?ns=students#/runs/details/7406981b-caac-4f40-8870-db932d750ea2) |
| `compile-fused` | 3228 | 5.9795 | 395.26 | [HTML](results/compile-fused.html) | [run](https://aviflow.k.avito.ru/_/pipeline/?ns=students#/runs/details/6db0308b-04a5-4b35-b6fb-7642816a8412) |
| `fp32-control` | 1226 | 6.4763 | 649.59 | [HTML](results/fp32-control.html) | [run](https://aviflow.k.avito.ru/_/pipeline/?ns=students#/runs/details/20360413-7f61-4d61-bd18-10a82d15c56a) |
| **`optimized`** | **3215** | **4.2284** | **68.61** | [HTML](results/optimized.html) | [run](https://aviflow.k.avito.ru/_/pipeline/?ns=students#/runs/details/a6cac7e6-84f8-49f8-ac52-aa1d606beeee) |

HTML-файлы являются сохранёнными Aviflow-артефактами.

## Анализ гиперпараметров

1. `batch-accum` и `large-batch` имеют одинаковый effective batch 16. Прямой
   batch 16 дал немного лучший `eval_loss`: 6.2739 против 6.3053, а также успел
   выполнить больше optimizer steps. На этой GPU прямой большой батч оказался
   немного эффективнее gradient accumulation.
2. При одинаковых batch, LR и scheduler профиль `compile-fused` оказался лучше
   `large-batch`: 5.9795 против 6.2739. Он выполнил 3228 вместо 2148 optimizer
   steps. Это показывает пользу комбинации `torch_compile`, fused AdamW и TF32
   в фиксированном временном бюджете. Поскольку три настройки изменялись вместе,
   по этим запускам нельзя строго разделить вклад каждой из них.
3. `fp32-control` выполнил только 1226 шагов и показал худшую perplexity 649.59.
   Для A100 использование BF16 существенно выгоднее FP32 по скорости и памяти.
4. Лучший результат дал `optimized`: LR `3e-4`, cosine scheduler, batch 16,
   fused AdamW, compile, BF16 и TF32. Относительно baseline eval loss снизился
   с 6.4439 до 4.2284, а perplexity — с 628.87 до 68.61 (примерно на 89%).
   Главным отличием от `compile-fused` являются более высокий LR и cosine
   scheduler; в коротком 15-минутном pretrain они дали намного более быстрое
   обучение.

## Evaluation и первичные генерации

Финальная evaluation выполнялась на первых 5000 примерах датасета. Для лучшего
профиля `optimized` зафиксировано:

```text
eval_loss = 4.228414058685303
perplexity = 68.60833704977294
global_step = 3215
```

После обучения модель получила три промпта. Полные результаты доступны в
[`results/optimized.html`](results/optimized.html). Примеры:

```text
Промпт: Москва — столица
Начало ответа: Москва — столица Нанси-Ланки.

Промпт: Искусственный интеллект помогает человеку
Начало ответа: Искусственный интеллект помогает человеку с его работой.

Промпт: Научное открытие начинается с
Начало ответа: Научное открытие начинается с другой, что происходит от
нескольких десятков слов в регионе и по крайней мере с севера на юг.
```
## Итог
 Лучшей оказалась конфигурация
`optimized`: batch 16, LR `3e-4`, cosine scheduler, fused AdamW,
`torch_compile`, BF16 и TF32. Она получила минимальные `eval_loss=4.2284` и
perplexity `68.61`, а её генерации уже внешне напоминают русскоязычные
энциклопедические статьи, хотя остаются фактически недостоверными и теряют тему
на длинном контексте.
