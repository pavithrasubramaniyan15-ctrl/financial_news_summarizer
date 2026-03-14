"""
app.py  –  Financial News Summarizer
=====================================
Flask web application.
Run:  python app.py
Then open http://127.0.0.1:5000
"""

import os
import json
import time
from flask import Flask, render_template, request, jsonify
from summarizer import FinancialNewsSummarizer
from evaluate  import compute_rouge

app = Flask(__name__)

# Shared summarizer instance (loaded once at startup)
_summarizer = None

def get_summarizer(num_sentences=5):
    global _summarizer
    if _summarizer is None or _summarizer.num_sentences != num_sentences:
        _summarizer = FinancialNewsSummarizer(num_sentences=num_sentences)
    return _summarizer


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text          = data["text"].strip()
    num_sentences = int(data.get("num_sentences", 5))
    show_rouge    = bool(data.get("show_rouge", False))

    if len(text.split()) < 30:
        return jsonify({"error": "Article is too short. Please paste a longer financial news article."}), 400

    start      = time.time()
    summarizer = get_summarizer(num_sentences)
    result     = summarizer.summarize(text)
    elapsed    = round(time.time() - start, 3)

    response = {
        "summary":              result["summary"],
        "selected_sentences":   result["selected_sentences"],
        "num_sentences":        result["num_sentences"],
        "total_sentences":      result["total_sentences"],
        "sentence_scores":      result.get("sentence_scores_detail", []),
        "processing_time_sec":  elapsed,
        "compression_ratio":    round(
            len(result["summary"].split()) / max(1, len(text.split())), 3
        ),
    }

    if show_rouge:
        try:
            rouge = compute_rouge(result["summary"], text)
            response["rouge"] = rouge
        except Exception as e:
            response["rouge_error"] = str(e)

    return jsonify(response)


@app.route("/sample", methods=["GET"])
def sample():
    """Return a sample financial news article for demo purposes."""
    samples = [
        {
            "title": "Apple Q2 Earnings",
            "text": (
                "Apple Inc. reported second-quarter results that exceeded Wall Street expectations, "
                "with revenue rising 5% year-over-year to $90.8 billion, driven by strong iPhone "
                "sales and record Services performance. The iPhone maker posted earnings per share "
                "of $1.53, beating the analyst consensus estimate of $1.50. Services revenue, which "
                "includes the App Store, Apple Music, iCloud, and Apple TV+, reached a new all-time "
                "high of $23.9 billion, growing 14% from the prior year period. CEO Tim Cook said "
                "the company continued to see robust demand despite global macroeconomic headwinds, "
                "particularly in developed markets. Greater China revenue disappointed investors, "
                "declining 8% to $16.4 billion amid intensifying competition from domestic smartphone "
                "makers Huawei and Xiaomi. Apple repurchased $23.5 billion of its own stock during the "
                "quarter and increased its quarterly dividend by 4% to $0.25 per share. The company "
                "provided guidance for the fiscal third quarter calling for revenue of $85 billion "
                "to $88 billion, largely in line with analyst forecasts. CFO Luca Maestri said gross "
                "margins are expected to remain strong at 45.5% to 46.5%. Shares of Apple rose 6.5% "
                "in after-hours trading on Thursday following the earnings release."
            )
        },
        {
            "title": "Federal Reserve Rate Decision",
            "text": (
                "The Federal Reserve left its benchmark interest rate unchanged at a 23-year high of "
                "5.25% to 5.50% at its May policy meeting, as policymakers said they need greater "
                "confidence that inflation is cooling before they begin cutting rates. Chair Jerome "
                "Powell acknowledged that progress on inflation has stalled in recent months, with "
                "the core Personal Consumption Expenditures price index running at 2.8%, still well "
                "above the central bank's 2% target. However, Powell pushed back against expectations "
                "of a rate hike, saying the next move is more likely to be a cut than an increase. "
                "Financial markets had briefly priced in the possibility of a hike earlier this month "
                "after a string of hot inflation readings. The Fed also announced it would slow the "
                "pace of quantitative tightening, reducing the monthly cap on Treasury security "
                "runoffs from $60 billion to $25 billion starting in June, a move designed to avoid "
                "undue strain on money markets. The 10-year Treasury yield fell 8 basis points to "
                "4.60% following the announcement, while the S&P 500 rose 0.9%. Goldman Sachs and "
                "JPMorgan both pushed back their forecasts for the first Fed rate cut to September, "
                "from an earlier projection of July."
            )
        },
    ]
    idx = int(request.args.get("idx", 0)) % len(samples)
    return jsonify(samples[idx])


if __name__ == "__main__":
    print("\n Financial News Summarizer is running!")
    print(" Open http://127.0.0.1:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
