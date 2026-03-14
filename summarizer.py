"""
summarizer.py  –  Financial News Summarizer
============================================
Core summarization engine with TWO approaches:

  1. EXTRACTIVE (default, no GPU needed)
     - Score each sentence using TF word frequencies
     - Boost sentences with financial keywords
     - Boost sentences with numbers / figures
     - Re-rank using cosine similarity between sentences
     - Pick top-N highest scoring sentences

  2. ABSTRACTIVE (optional, requires sentence-transformers)
     - Encode sentences with a pretrained SBERT model
     - Cluster similar sentences (MMR diversity)
     - Return the most representative sentences

Both modes return a ranked list of original sentences from the article.
"""

import re
import math
import numpy as np
from nltk.tokenize import word_tokenize

from preprocessing import FinancialTextPreprocessor, FINANCIAL_KEYWORDS


class FinancialNewsSummarizer:
    """
    Bloomberg-style extractive summarizer for financial news.

    Usage:
        summarizer = FinancialNewsSummarizer(num_sentences=5)
        result = summarizer.summarize(article_text)
        print(result["summary"])
    """

    def __init__(self, num_sentences: int = 5, use_semantic: bool = False):
        """
        Args:
            num_sentences: number of sentences in the output summary
            use_semantic : if True, use SBERT sentence embeddings for
                           similarity (slower but more accurate)
        """
        self.num_sentences = num_sentences
        self.use_semantic  = use_semantic
        self.preprocessor  = FinancialTextPreprocessor()
        self._sbert_model  = None   # lazy-loaded only if needed

    # ── Public API ──────────────────────────────────────────────────────────

    def summarize(self, text: str) -> dict:
        """
        Summarize a financial news article.

        Args:
            text: full article as a plain string

        Returns:
            dict:
              summary          – the generated summary (string)
              selected_sentences – list of chosen sentences
              scores           – sentence score array
              num_sentences    – actual sentences returned
        """
        # Step 1 – preprocess
        data = self.preprocessor.preprocess(text)
        sentences  = data["sentences"]
        word_freq  = data["word_frequencies"]
        fin_scores = data["financial_scores"]

        if not sentences:
            return {"summary": text, "selected_sentences": [text],
                    "scores": [], "num_sentences": 1}

        # Clamp to available sentences
        n = min(self.num_sentences, len(sentences))

        # Step 2 – score sentences
        tf_scores = self._score_sentences_tf(sentences, word_freq, fin_scores)

        # Step 3 – re-rank with similarity (optional SBERT, else TF-IDF cosine)
        if self.use_semantic:
            final_scores = self._semantic_rerank(sentences, tf_scores)
        else:
            final_scores = self._cosine_rerank(sentences, tf_scores)

        # Step 4 – pick top-N, preserve original article order
        ranked_idx = np.argsort(final_scores)[::-1][:n]
        ordered_idx = sorted(ranked_idx)               # keep reading order
        selected   = [sentences[i] for i in ordered_idx]
        summary    = " ".join(selected)

        return {
            "summary":            summary,
            "selected_sentences": selected,
            "scores":             final_scores.tolist(),
            "sentence_scores_detail": [
                {"sentence": s[:80]+"…" if len(s)>80 else s,
                 "score": round(float(final_scores[i]), 4)}
                for i,s in enumerate(sentences)
            ],
            "num_sentences": n,
            "total_sentences": len(sentences),
        }

    # ── Scoring ─────────────────────────────────────────────────────────────

    def _score_sentences_tf(self, sentences, word_freq, fin_scores):
        """
        Score each sentence by summing the TF-scores of its words,
        then apply bonuses for:
          - financial keyword density
          - presence of numbers / monetary figures
          - sentence position (first sentences score higher)
        """
        scores = np.zeros(len(sentences))

        for i, sentence in enumerate(sentences):
            words = word_tokenize(sentence.lower())
            if not words:
                continue

            # Base TF score
            tf_score = sum(word_freq.get(w, 0) for w in words) / len(words)

            # Financial keyword bonus (0–0.3 additive)
            fin_bonus = fin_scores[i] * 0.3

            # Number / figure bonus: sentences with $ amounts score higher
            num_pattern = r"\$[\d,]+|\d+\.?\d*\s*(?:billion|million|trillion|%|percent)"
            num_bonus   = 0.15 if re.search(num_pattern, sentence, re.I) else 0

            # Position bonus: first 20% of article sentences score +0.1
            position_bonus = 0.1 if i < max(1, len(sentences) * 0.2) else 0

            # Length penalty: very short or very long sentences score lower
            word_count = len(words)
            if word_count < 8:
                length_factor = 0.7
            elif word_count > 50:
                length_factor = 0.85
            else:
                length_factor = 1.0

            scores[i] = (tf_score + fin_bonus + num_bonus + position_bonus) * length_factor

        return scores

    def _cosine_rerank(self, sentences, tf_scores):
        """
        Re-rank using TF-IDF cosine similarity so that sentences covering
        DIFFERENT aspects of the article are preferred (diversity).

        Approach: MMR (Maximal Marginal Relevance) – balance relevance vs. novelty.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        if len(sentences) == 1:
            return tf_scores

        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_mat  = vectorizer.fit_transform(sentences)
            sim_matrix = cosine_similarity(tfidf_mat)
        except Exception:
            return tf_scores

        # MMR: score[i] = lambda * relevance[i] - (1-lambda) * max_sim_to_selected[i]
        lmbda      = 0.7
        n          = len(sentences)
        selected   = []
        mmr_scores = tf_scores.copy()

        for _ in range(n):
            if selected:
                # Penalise redundancy
                for j in range(n):
                    max_sim = max(sim_matrix[j][s] for s in selected)
                    mmr_scores[j] = (lmbda * tf_scores[j]
                                     - (1 - lmbda) * max_sim)
            best = int(np.argmax(mmr_scores))
            selected.append(best)
            mmr_scores[best] = -999   # mark as used

        # Convert rank order back to a score array
        final = np.zeros(n)
        for rank, idx in enumerate(selected):
            final[idx] = (n - rank) / n
        return final

    def _semantic_rerank(self, sentences, tf_scores):
        """
        Use SBERT embeddings for semantic similarity (higher quality).
        Falls back to cosine_rerank if sentence-transformers is unavailable.
        """
        try:
            from sentence_transformers import SentenceTransformer
            import torch

            if self._sbert_model is None:
                print("Loading SBERT model (first run only)…")
                self._sbert_model = SentenceTransformer(
                    "all-MiniLM-L6-v2"   # small, fast, good quality
                )

            embeddings = self._sbert_model.encode(sentences,
                                                   convert_to_numpy=True)
            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(embeddings)

            # Same MMR as above but with semantic similarities
            lmbda    = 0.7
            n        = len(sentences)
            selected = []
            mmr      = tf_scores.copy()

            for _ in range(n):
                if selected:
                    for j in range(n):
                        max_sim = max(sim_matrix[j][s] for s in selected)
                        mmr[j]  = lmbda * tf_scores[j] - (1-lmbda)*max_sim
                best = int(np.argmax(mmr))
                selected.append(best)
                mmr[best] = -999

            final = np.zeros(n)
            for rank, idx in enumerate(selected):
                final[idx] = (n - rank) / n
            return final

        except ImportError:
            print("sentence-transformers not available – using TF-IDF cosine instead.")
            return self._cosine_rerank(sentences, tf_scores)


# ── Convenience wrapper ──────────────────────────────────────────────────────

def summarize_article(text: str, num_sentences: int = 5,
                       use_semantic: bool = False) -> str:
    """One-liner wrapper – returns just the summary string."""
    s = FinancialNewsSummarizer(num_sentences=num_sentences,
                                 use_semantic=use_semantic)
    return s.summarize(text)["summary"]


if __name__ == "__main__":
    article = """
    Apple Inc. reported record quarterly earnings on Thursday, with revenue surging
    8% year-over-year to $119.6 billion, beating Wall Street estimates of $117.9 billion.
    The iPhone maker's net income rose to $33.9 billion, or $2.18 per diluted share,
    compared with $30.0 billion a year ago. Services revenue, which includes the App Store,
    Apple Music, and iCloud, climbed 16% to $23.1 billion, marking another all-time high.
    CEO Tim Cook said the company saw strong demand across all product categories and
    geographies, with particular strength in emerging markets. However, Mac and iPad sales
    disappointed analysts, falling short of consensus estimates. The company also announced
    a $110 billion share buyback programme, the largest in its history, alongside a 4%
    increase in its quarterly dividend to $0.25 per share. Apple's stock rose 6% in
    after-hours trading following the announcement. CFO Luca Maestri provided guidance for
    the next quarter, projecting revenue between $85 billion and $88 billion, slightly
    above analyst expectations of $84.7 billion. The company said it expects gross margins
    to remain between 45.5% and 46.5%, reflecting strong pricing power despite ongoing
    supply chain pressures. Analysts at Goldman Sachs maintained their Buy rating on the
    stock and raised their 12-month price target from $210 to $230.
    """

    summarizer = FinancialNewsSummarizer(num_sentences=4)
    result     = summarizer.summarize(article)
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(result["summary"])
    print(f"\nUsed {result['num_sentences']} of {result['total_sentences']} sentences")
