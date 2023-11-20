# Cantonese-Mandarin Machine Translation

## Package managing

If using conda:

```
conda create -n 2590-project poetry
conda activate 2590-project
poetry install --no-root
```

If not using conda, first install poetry, then:

```
poetry install --no-root
```

When installing new packages to the virtual environment, first add to `pyproject.toml`, then:

```
poetry lock
poetry install --no-root
```

## Evaluation script and results

Use `evaluate.py -h` to see the options. The script takes in a plain text output file, whose rows are the model translations, and output BLEU and ChrF++ scores.

We have 2 directions(Mandarin-Cantonese, Cantonese-Mandarin), 2 data sources(Main, Tatoeba), 2 splits(validation, test), 2 metrics, so 16 values for each method.

Mandarin to Cantonese:

| Method(validation/test) | BLEU(main)    | BLEU(tatoeba) | ChrF++(main)  | ChrF++(tatoeba) |
| ----------------------- | ------------- | ------------- | ------------- | --------------- |
| Naive baseline          | 12.356/13.103 | 21.987/24.761 | 10.872/11.373 | 16.195/16.645   |
| Existing work           |               |               |               |                 |
| 0-shot                  |               |               |               |                 |
| 5-shot                  |               |               |               |                 |
| Finetuned               |               |               |               |                 |

Cantonese to Mandarin:

| Method(validation/test) | BLEU(main)    | BLEU(tatoeba) | ChrF++(main)  | ChrF++(tatoeba) |
| ----------------------- | ------------- | ------------- | ------------- | --------------- |
| Naive baseline          | 12.437/13.181 | 21.974/24.711 | 10.617/11.136 | 16.450/16.825   |
| Existing work           |               |               |               |                 |
| 0-shot                  |               |               |               |                 |
| 5-shot                  |               |               |               |                 |
| Finetuned               |               |               |               |                 |
