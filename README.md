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
| 0-shot                  | 12.898/13.097 | 23.650/24.073 |       /14.149 | 26.083/27.214   |
| 5-shot                  |               |               |               |                 |
| Finetuned               | 37.738/35.371 | 49.522/44.583 | 28.841/26.197 | 38.884/35.274   |

Cantonese to Mandarin:

| Method(validation/test) | BLEU(main)    | BLEU(tatoeba) | ChrF++(main)  | ChrF++(tatoeba) |
| ----------------------- | ------------- | ------------- | ------------- | --------------- |
| Naive baseline          | 12.437/13.181 | 21.974/24.711 | 10.617/11.136 | 16.450/16.825   |
| Existing work           |               |               |               |                 |
| 0-shot                  | 8.417/8.646   | 14.970/13.581 |       /10.654 | 18.253/18.001   |
| 5-shot                  |               |               |               |                 |
| Finetuned               | 36.469/36.553 | 44.444/47.719 | 27.028/27.471 | 31.874/37.925   |
