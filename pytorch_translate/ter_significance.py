#!/usr/bin/env python3

import argparse
from typing import List, NamedTuple, Optional

import numpy as np
import pandas as pd
import sacrebleu


DEFAULT_TOKENIZER = 'none'


def get_sufficient_stats(
    translations: List[str], references: List[str]
) -> pd.DataFrame:
    assert len(translations) == len(references), (
        f"There are {len(translations)} translated sentences "
        f"but {len(references)} reference sentences"
    )
    sufficient_stats: List[List[int]] = []
    for sentence, ref in zip(translations, references):
        sentence_ter = sacrebleu.sentence_ter(
            hypothesis=sentence,
            references=[ref],
            case_sensitive=True,
            normalized=True)
        sufficient_stats.append([sentence_ter.score, sentence_ter.num_edits, sentence_ter.ref_length])
    return pd.DataFrame(
        sufficient_stats,
        columns=[
            "score",
            "num_edits",
            "ref_length",
        ],
    )


class PairedBootstrapOutput(NamedTuple):
    baseline_ter: float
    new_ter: float
    num_samples: int
    # Number of samples where the baseline was better than the new.
    baseline_better: int
    # Number of samples where the baseline and new had identical TER score.
    num_equal: int
    # Number of samples where the new was better than baseline.
    new_better: int


def compute_ter_from_stats(stats: pd.DataFrame):
    return stats.score.mean(axis=0)


def paired_bootstrap_resample(
    baseline_stats: pd.DataFrame,
    new_stats: pd.DataFrame,
    num_samples: int = 1000,
    sample_size: Optional[int] = None,
) -> PairedBootstrapOutput:
    """
    From http://aclweb.org/anthology/W04-3250
    Statistical significance tests for machine translation evaluation (Koehn, 2004)
    """
    assert len(baseline_stats) == len(new_stats), (
        f"Length mismatch - baseline has {len(baseline_stats)} lines "
        f"while new has {len(new_stats)} lines."
    )
    num_sentences = len(baseline_stats)
    if not sample_size:
        # Defaults to sampling new corpora of the same size as the original.
        # This is not identical to the original corpus since we are sampling
        # with replacement.
        sample_size = num_sentences
    indices = np.random.randint(
        low=0, high=num_sentences, size=(num_samples, sample_size)
    )

    baseline_better: int = 0
    new_better: int = 0
    num_equal: int = 0
    for index in indices:
        baseline_ter = compute_ter_from_stats(baseline_stats.iloc[index])
        new_ter = compute_ter_from_stats(new_stats.iloc[index])
        if new_ter < baseline_ter:
            new_better += 1
        elif baseline_ter < new_ter:
            baseline_better += 1
        else:
            # If the baseline corpus and new corpus are identical, this
            # degenerate case may occur.
            num_equal += 1

    return PairedBootstrapOutput(
        baseline_ter=compute_ter_from_stats(baseline_stats),
        new_ter=compute_ter_from_stats(new_stats),
        num_samples=num_samples,
        baseline_better=baseline_better,
        num_equal=num_equal,
        new_better=new_better,
    )


def paired_bootstrap_resample_from_files(
    reference_file: str,
    baseline_file: str,
    new_file: str,
    num_samples: int = 1000,
    sample_size: Optional[int] = None,
) -> PairedBootstrapOutput:
    with open(reference_file, "r") as f:
        references: List[str] = [line for line in f]

    with open(baseline_file, "r") as f:
        baseline_translations: List[str] = [line for line in f]
    baseline_stats: pd.DataFrame = get_sufficient_stats(
        translations=baseline_translations, references=references
    )

    with open(new_file, "r") as f:
        new_translations: List[str] = [line for line in f]
    new_stats: pd.DataFrame = get_sufficient_stats(
        translations=new_translations, references=references
    )

    return paired_bootstrap_resample(
        baseline_stats=baseline_stats,
        new_stats=new_stats,
        num_samples=num_samples,
        sample_size=sample_size,
    )


def main():
    parser = argparse.ArgumentParser()
    tokenization_warning = "Sentences are tokenized with Tercom tokenizer.",
    parser.add_argument(
        "--reference-file",
        type=str,
        required=True,
        help=f"Text file containing reference sentences. {tokenization_warning}",
    )
    parser.add_argument(
        "--baseline-file",
        type=str,
        required=True,
        help=f"Text file containing sentences translated by baseline system. {tokenization_warning}",
    )
    parser.add_argument(
        "--new-file",
        type=str,
        required=True,
        help=f"Text file containing sentences translated by new system. {tokenization_warning}",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        required=False,
        default=1000,
        help="Number of comparisons to be executed.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        required=False,
        help="Number of sentences sampled for each comparison.",
    )
    args = parser.parse_args()

    output = paired_bootstrap_resample_from_files(
        reference_file=args.reference_file,
        baseline_file=args.baseline_file,
        new_file=args.new_file,
        num_samples=args.num_samples,
        sample_size=getattr(args, "sample_size", None),
    )

    print(f"Baseline TER: {output.baseline_ter:.4f}")
    print(f"New TER: {output.new_ter:.4f}")
    print(f"TER delta: {output.new_ter - output.baseline_ter:.4f} ")
    print(
        f"Baseline better confidence: {output.baseline_better / output.num_samples:.2%}"
    )
    print(f"New better confidence: {output.new_better / output.num_samples:.2%}")


if __name__ == "__main__":
    main()
