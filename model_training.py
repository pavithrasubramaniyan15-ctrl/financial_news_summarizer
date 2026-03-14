"""
model_training.py
=================
Financial News Summarizer - Model Training

Two modes:
  1. tfidf      : Fit TF-IDF vectorizer on your corpus (beginner-friendly, no GPU)
  2. transformer: Fine-tune BART on labeled (article, summary) pairs (GPU recommended)
  3. stats      : Print dataset statistics

Usage:
    python model_training.py --mode tfidf --dataset dataset/articles.csv
    python model_training.py --mode stats --dataset dataset/articles.csv
"""

import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm


def train_tfidf_model(dataset_path, output_path="model/tfidf_vocab.pkl"):
    """Fit TF-IDF on financial news corpus for better domain-specific scoring."""
    import pickle
    from sklearn.feature_extraction.text import TfidfVectorizer
    from preprocessing import TextPreprocessor

    print(f"Loading dataset from {dataset_path} ...")
    df = pd.read_csv(dataset_path)

    text_col = None
    for col in ["article", "text", "content", "body"]:
        if col in df.columns:
            text_col = col
            break
    if not text_col:
        raise ValueError(f"No text column found. Columns: {list(df.columns)}")

    df = df.dropna(subset=[text_col])
    texts = df[text_col].tolist()
    print(f"  Loaded {len(texts)} articles.")

    preprocessor = TextPreprocessor()
    processed_texts = []
    print("  Preprocessing articles...")
    for text in tqdm(texts[:5000]):
        result = preprocessor.preprocess_article(str(text))
        combined = " ".join(" ".join(t) for t in result["processed_sentences"])
        if combined.strip():
            processed_texts.append(combined)

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                                 min_df=2, sublinear_tf=True)
    vectorizer.fit(processed_texts)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"  TF-IDF vectorizer saved to '{output_path}'")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    return vectorizer


def dataset_statistics(dataset_path):
    """Print useful statistics about the dataset."""
    df = pd.read_csv(dataset_path)
    text_col = None
    for col in ["article", "text", "content", "body"]:
        if col in df.columns:
            text_col = col
            break
    if not text_col:
        print("Could not find text column.")
        return

    df = df.dropna(subset=[text_col])
    lengths = df[text_col].apply(lambda x: len(str(x).split()))

    print("\nDataset Statistics")
    print("=" * 40)
    print(f"  Total articles    : {len(df)}")
    print(f"  Avg word count    : {lengths.mean():.0f}")
    print(f"  Median word count : {lengths.median():.0f}")
    print(f"  Min word count    : {lengths.min()}")
    print(f"  Max word count    : {lengths.max()}")

    if "summary" in df.columns:
        slen = df["summary"].dropna().apply(lambda x: len(str(x).split()))
        print(f"  Avg summary length: {slen.mean():.0f} words")
        comp = 1 - (slen.mean() / lengths.mean())
        print(f"  Avg compression   : {comp*100:.1f}%")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tfidf", "stats"], default="tfidf")
    parser.add_argument("--dataset", default="dataset/articles.csv")
    parser.add_argument("--output",  default="model/tfidf_vocab.pkl")
    args = parser.parse_args()

    if args.mode == "tfidf":
        train_tfidf_model(args.dataset, args.output)
    elif args.mode == "stats":
        dataset_statistics(args.dataset)
