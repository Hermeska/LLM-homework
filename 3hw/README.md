# Домашняя работа 3: SFT для function calling

Основное решение находится в `sft_function_calling_hw.ipynb`. Notebook обучает
все веса `Qwen/Qwen2.5-0.5B-Instruct`, сравнивает base и SFT-модели и запускает
агентный цикл с симулятором умного дома.

## Запуск в Aviflow

```bash
cd ~/domashka/LLM-homework/3hw
~/.venvs/evolutionrl-aviflow/bin/python aviflow_pipeline.py \
  --namespace students \
  --run-name qwen25-sft-function-calling-v1
```

Pipeline использует одну GPU и возвращает архив без весов модели. В архив входят:

- выполненный `sft_function_calling_hw.ipynb` со всеми выводами;
- `outputs/eval_results.csv`.

Каталог `outputs/full_sft/` содержит локальные веса и исключён из Git.

