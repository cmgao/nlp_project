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

## ALMA fine-tuning

Code and documentation in the ALMA submodule.

Huggingface: https://huggingface.co/superaidesu/cantonese-alma-2-7b-oasst-v1-lora

## Evaluation script and results

Use `evaluate.py -h` to see the options. The script takes in a plain text output file, whose rows are the model translations, and output BLEU and ChrF++ scores.

We have 2 directions(Mandarin-Cantonese, Cantonese-Mandarin), 2 data sources(Main, Tatoeba), 2 splits(validation, test), 2 metrics, so 16 values for each method.

Mandarin to Cantonese:

| Method(validation/test) | BLEU(main)    | BLEU(tatoeba) | ChrF++(main)  | ChrF++(tatoeba) |
| ----------------------- | ------------- | ------------- | ------------- | --------------- |
| Naive baseline          | 12.356/13.103 | 21.987/24.761 | 10.872/11.373 | 16.195/16.645   |
| Existing work           |      ./24.941 |      ./36.878 |      ./19.056 |      ./24.717   |
| 0-shot                  | 10.458/11.217 | 20.286/20.977 | 9.824/10.368  | 15.641/16.085   |
| 5-shot                  | 9.825/10.346  | 18.119/18.895 | 8.946/9.059   | 13.818/14.584   |
| Finetuned               | 37.738/35.371 | 49.522/44.583 | 28.841/26.197 | 38.884/35.274   |

Cantonese to Mandarin:

| Method(validation/test) | BLEU(main)    | BLEU(tatoeba) | ChrF++(main)  | ChrF++(tatoeba) |
| ----------------------- | ------------- | ------------- | ------------- | --------------- |
| Naive baseline          | 12.437/13.181 | 21.974/24.711 | 10.617/11.136 | 16.450/16.825   |
| Existing work           |      ./16.534 |      ./28.999 |      ./13.259 |      ./20.304   |
| 0-shot                  | 7.749/13.208  | 13.687/13.208 | 7.177/10.995  | 11.061/10.995   |
| 5-shot                  | 5.849/6.723   | 10.582/12.332 | 6.552/6.832   | 9.520/10.064    |
| Finetuned               | 36.469/36.553 | 44.444/47.719 | 27.028/27.471 | 31.874/37.925   |
