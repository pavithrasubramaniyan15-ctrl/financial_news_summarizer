"""
evaluate.py  –  Financial News Summarizer
==========================================
Evaluates summary quality using ROUGE metrics.

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures:
  ROUGE-1 : unigram overlap between summary and reference
  ROUGE-2 : bigram  overlap
  ROUGE-L : longest common subsequence

Usage:
    python evaluate.py
    python evaluate.py --csv dataset/financial_news.csv --col article --n 20
"""

import argparse
import json
from pathlib import Path

from rouge_score import rouge_scorer


def compute_rouge(hypothesis: str, reference: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L between two texts.

    Args:
        hypothesis : the generated summary
        reference  : the reference / gold summary (or first N% of article)

    Returns:
        dict with precision, recall, f1 for each ROUGE type
    """
    scorer  = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"],
                                        use_stemmer=True)
    scores  = scorer.score(reference, hypothesis)
    result  = {}
    for key, val in scores.items():
        result[key] = {
            "precision": round(val.precision, 4),
            "recall":    round(val.recall,    4),
            "f1":        round(val.fmeasure,  4),
        }
    return result


def evaluate_on_dataset(articles: list, summarizer,
                         compression_ratio: float = 0.3) -> dict:
    """
    Run summarizer over a list of articles and compute average ROUGE scores.

    We use the FIRST 30% of each article as a pseudo-reference
    (a common practice when gold summaries are unavailable).

    Args:
        articles          : list of article strings
        summarizer        : FinancialNewsSummarizer instance
        compression_ratio : fraction of article to treat as reference

    Returns:
        dict of average ROUGE scores across all articles
    """
    from nltk.tokenize import sent_tokenize

    all_scores = {"rouge1":{"precision":[],"recall":[],"f1":[]},
                  "rouge2":{"precision":[],"recall":[],"f1":[]},
                  "rougeL":{"precision":[],"recall":[],"f1":[]}}

    for i, article in enumerate(articles):
        if len(article.split()) < 50:
            continue   # too short to evaluate

        # Build pseudo-reference: first ~30% of sentences
        sents      = sent_tokenize(article)
        ref_n      = max(1, int(len(sents) * compression_ratio))
        reference  = " ".join(sents[:ref_n])

        result     = summarizer.summarize(article)
        hypothesis = result["summary"]

        rouge      = compute_rouge(hypothesis, reference)

        for metric in all_scores:
            for subkey in ["precision","recall","f1"]:
                all_scores[metric][subkey].append(rouge[metric][subkey])

        if (i+1) % 10 == 0:
            print(f"  Evaluated {i+1}/{len(articles)} articles…")

    # Average
    import numpy as np
    avg = {}
    for metric in all_scores:
        avg[metric] = {
            k: round(float(np.mean(v)), 4)
            for k,v in all_scores[metric].items()
            if v
        }
    return avg


def print_rouge_table(scores: dict):
    """Pretty-print ROUGE scores as a table."""
    print("\n" + "="*52)
    print(f"{'METRIC':<12} {'PRECISION':>10} {'RECALL':>10} {'F1':>10}")
    print("="*52)
    for metric, vals in scores.items():
        print(f"{metric.upper():<12} {vals.get('precision',0):>10.4f} "
              f"{vals.get('recall',0):>10.4f} {vals.get('f1',0):>10.4f}")
    print("="*52)


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate summarizer with ROUGE")
    parser.add_argument("--csv", default=None, help="Path to financial news CSV")
    parser.add_argument("--col", default="article", help="Text column name in CSV")
    parser.add_argument("--n",   default=20,  type=int, help="Number of articles")
    parser.add_argument("--sentences", default=5, type=int,
                        help="Summary length in sentences")
    args = parser.parse_args()

    from summarizer import FinancialNewsSummarizer
    summarizer = FinancialNewsSummarizer(num_sentences=args.sentences)

    if args.csv:
        from preprocessing import load_financial_dataset
        articles = load_financial_dataset(args.csv, args.col, args.n)
        print(f"\nEvaluating on {len(articles)} articles…")
        avg = evaluate_on_dataset(articles, summarizer)
        print_rouge_table(avg)
        out = Path("model/rouge_scores.json")
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(avg, indent=2))
        print(f"Scores saved to {out}")
    else:
        # Demo on a single hard-coded article
        demo = (
            "JPMorgan Chase reported second-quarter profit of $14.5 billion, "
            "or $4.75 per share, topping analyst expectations of $4.29 per share. "
            "Revenue climbed 23% to $42.4 billion, lifted by higher interest income. "
            "The bank's net interest income surged 44% to $21.9 billion as the Federal "
            "Reserve's aggressive rate-hiking cycle boosted lending margins. CEO Jamie "
            "Dimon warned, however, that storm clouds remain on the horizon, citing "
            "persistent inflation, rising geopolitical tensions, and the lagged effects "
            "of quantitative tightening. The investment banking division saw fees drop "
            "6% as deal-making activity remained subdued amid market uncertainty. "
            "Credit card spending rose 14% year-over-year but the bank increased loan "
            "loss provisions by $2.9 billion, reflecting caution about consumer credit. "
            "JPMorgan raised its full-year net interest income guidance to $87 billion, "
            "up from an earlier estimate of $84 billion, sending shares up 4% in "
            "pre-market trading on Friday."
        )
        result = summarizer.summarize(demo)
        rouge  = compute_rouge(result["summary"], demo)
        print("\nDemo article summary:")
        print(result["summary"])
        print_rouge_table(rouge)
