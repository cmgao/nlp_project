import argparse
import pandas as pd
from typing import Sequence, Callable
from sacrebleu.metrics import BLEU, CHRF


def evaluate(ref: Sequence[str], pred: Sequence[str]) -> dict:
    """
    Evaluate the predictions against the reference using BLEU and chrF++.
    :param ref: list of reference sentences
    :param pred: list of predicted sentences
    :return: dictionary of scores
    """
    print("Reference:")
    print(ref[:5])
    print("Prediction:")
    print(pred[:5])

    bleu = BLEU(tokenize="zh")
    chrf = CHRF(word_order=2)

    bleu_score = bleu.corpus_score(pred, [ref]).score
    chrf_score = chrf.corpus_score(pred, [ref]).score

    return {"bleu": bleu_score, "chrf++": chrf_score}


def dummy_pred(ref: pd.DataFrame, target: str) -> Sequence[str]:
    """
    Use source text as naive baseline.
    """
    return ref.iloc[:, 1].tolist() if target == "yue" else ref.iloc[:, 0].tolist()


def evaluation_driver(input: str | Callable | None, split="val", domain="main", target="yue"):
    """
    :param input: If None, use source text as prediction.
                If callable, use it to generate predictions. It should take a list of source text and target language specifier.
                Otherwise, read from file.
    :param split: test or val
    :param domain: main or tatoeba
    :param target: yue or cmn
    """
    if split == "test":
        if domain == "main":
            ref = pd.read_csv("data/test_cleaned_parallel_sentences.txt", header=None).iloc[:, -2:]
        else:
            ref = pd.read_csv("data/test_tatoeba_sentences.txt", header=None)
    elif domain == "main":
        ref = pd.read_csv("data/valid_cleaned_parallel_sentences.txt", header=None).iloc[:, -2:]
    else:
        ref = pd.read_csv("data/valid_tatoeba_sentences.txt", header=None)

    if input is None:
        pred = dummy_pred(ref, target)
    elif callable(input):
        pred_input = dummy_pred(ref, target)
        pred = input(pred_input, target)
    else:
        with open(input, "r") as f:
            pred = f.readlines()

    ref_list = ref.iloc[:, 0].tolist() if target == "yue" else ref.iloc[:, 1].tolist()

    scores = evaluate(ref_list, pred)

    print(scores)

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', dest='split', action='store_const',
                    const="test", default="val",
                    help='Evaluate on test set instead of validation test')
    parser.add_argument('--input', help='Path to input file (contains predictions)', default=None)
    parser.add_argument("--domain", help='which data source to use (main or tatoeba)', choices=["main", "tatoeba"], required=True)
    parser.add_argument('--target', help='Target language (yue or cmn)', choices=["yue", "cmn"], required=True)

    args = parser.parse_args()

    evaluation_driver(args.input, args.split, args.domain, args.target)