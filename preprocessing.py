"""
preprocessing.py  –  Financial News Summarizer
===============================================
Handles all NLP preprocessing:
  1. Text cleaning
  2. Sentence & word tokenisation
  3. Stopword removal
  4. Financial keyword boosting
  5. TF-style word-frequency computation
"""

import re
import string

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer

# Download NLTK data once
for _r in ["punkt", "stopwords", "averaged_perceptron_tagger", "punkt_tab"]:
    try:
        nltk.download(_r, quiet=True)
    except Exception:
        pass

# Financial domain keywords – given extra weight during scoring
FINANCIAL_KEYWORDS = {
    "stock","market","price","share","equity","index","indices",
    "bull","bear","rally","crash","correction","volatility",
    "earnings","revenue","profit","loss","dividend","acquisition",
    "merger","ipo","bankruptcy","restructuring","layoff",
    "gdp","inflation","interest","rate","federal","reserve","fed",
    "monetary","fiscal","recession","growth","unemployment",
    "bond","yield","treasury","crypto","bitcoin","commodity",
    "oil","gold","currency","forex","derivative","futures",
    "bank","fund","hedge","investor","analyst","ceo","cfo",
    "quarter","annual","forecast","guidance","outlook",
    "billion","million","trillion","percent",
}


class FinancialTextPreprocessor:
    """
    Cleans and tokenises financial news articles.

    Usage:
        pre = FinancialTextPreprocessor()
        result = pre.preprocess(article_text)
        # result["sentences"]        -> list of sentence strings
        # result["word_frequencies"] -> {word: normalised_freq}
        # result["financial_scores"] -> per-sentence keyword density
    """

    def __init__(self, language="english"):
        self.language   = language
        self.stemmer    = PorterStemmer()
        self.stop_words = set(stopwords.words(language))
        # Preserve financially meaningful words NLTK would strip
        self.stop_words -= {"up","down","high","low","above","below",
                            "under","over","new","most","more","not","no"}

    # ── public ──────────────────────────────────────────────────────────────
    def preprocess(self, text: str) -> dict:
        """Full pipeline: text -> sentences + word frequencies + scores."""
        cleaned   = self._clean_text(text)
        sentences = self._tokenize_sentences(cleaned)
        word_freq = self._compute_word_frequencies(cleaned)
        fin_scores = self._financial_keyword_scores(sentences)
        lengths   = [len(word_tokenize(s)) for s in sentences]
        return {
            "original_text":    text,
            "cleaned_text":     cleaned,
            "sentences":        sentences,
            "word_frequencies": word_freq,
            "financial_scores": fin_scores,
            "sentence_lengths": lengths,
        }

    # ── private ──────────────────────────────────────────────────────────────
    def _clean_text(self, text):
        text = re.sub(r"\s+",      " ",  text).strip()
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"\S+@\S+\.\S+",          "", text)
        text = re.sub(r"<[^>]+>",               "", text)
        text = text.replace("$"," $ ").replace("%"," % ")
        text = re.sub(r"[^a-zA-Z0-9\s.,!?;:$%\-()]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _tokenize_sentences(self, text):
        return [s.strip() for s in sent_tokenize(text)
                if len(word_tokenize(s)) >= 5]

    def _compute_word_frequencies(self, text):
        words    = word_tokenize(text.lower())
        filtered = [w for w in words
                    if w not in string.punctuation
                    and w not in self.stop_words
                    and len(w) > 1]
        freq = {}
        for w in filtered:
            freq[w] = freq.get(w, 0) + 1
        # Boost financial keywords
        for w in freq:
            if w in FINANCIAL_KEYWORDS:
                freq[w] *= 1.5
        if freq:
            mx = max(freq.values())
            freq = {w: round(f/mx, 4) for w,f in freq.items()}
        return freq

    def _financial_keyword_scores(self, sentences):
        scores = []
        for s in sentences:
            words = word_tokenize(s.lower())
            if not words:
                scores.append(0.0); continue
            kw = sum(1 for w in words if w in FINANCIAL_KEYWORDS)
            scores.append(round(kw/len(words), 4))
        return scores


def load_financial_dataset(csv_path, text_column="article", n_samples=100):
    """
    Load financial news articles from a CSV file.

    Compatible Kaggle datasets:
      * All-the-news  (text_column='content')
      * Financial PhraseBank / Reuters financial news

    Args:
        csv_path   : path to CSV
        text_column: column containing article text
        n_samples  : how many rows to load (None = all)
    Returns:
        list of article strings
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found. "
                         f"Available: {list(df.columns)}")
    df = df[df[text_column].notna()].reset_index(drop=True)
    if n_samples:
        df = df.head(n_samples)
    articles = df[text_column].tolist()
    print(f"Loaded {len(articles)} articles from '{csv_path}'")
    return articles


if __name__ == "__main__":
    sample = (
        "The Federal Reserve raised interest rates by 25 basis points on Wednesday, "
        "marking the tenth consecutive increase as policymakers battle inflation. "
        "The S&P 500 fell 1.2% following the announcement, while the 10-year Treasury "
        "yield climbed to 4.8%. Fed Chair Jerome Powell stated the central bank remains "
        "committed to its 2% inflation target even as GDP growth shows signs of slowing. "
        "Major banks including JPMorgan and Goldman Sachs revised 2024 earnings forecasts "
        "downward, citing tighter credit conditions. Oil prices dropped 3% to $78 per barrel "
        "on fears of reduced global demand, while gold rose 0.5% as investors sought safety."
    )
    pre    = FinancialTextPreprocessor()
    result = pre.preprocess(sample)
    print(f"Sentences : {len(result['sentences'])}")
    print(f"Unique words : {len(result['word_frequencies'])}")
    print("Top-10 words:")
    for w,f in sorted(result["word_frequencies"].items(), key=lambda x:-x[1])[:10]:
        print(f"  {w:<20} {f:.4f}")
