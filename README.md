# 📰 Financial News Summarizer
### Bloomberg-style AI Briefings using NLP

A complete NLP project that summarizes long financial news articles into
concise briefings using extractive summarization, TF-IDF scoring, semantic
similarity, and MMR-based diversity selection.

---

## 📁 Project Structure

```
financial_news_summarizer/
│
├── dataset/                    ← Place your Kaggle CSV here
│   └── financial_news.csv
│
├── model/                      ← Auto-generated after training
│   ├── tfidf_vectorizer.pkl
│   ├── tfidf_stats.json
│   └── rouge_scores.json
│
├── templates/
│   └── index.html              ← Web UI
│
├── preprocessing.py            ← Text cleaning, tokenization, TF scoring
├── summarizer.py               ← Core summarization engine (Extractive + MMR)
├── model_training.py           ← TF-IDF corpus training + Transformer fine-tuning
├── evaluate.py                 ← ROUGE-1, ROUGE-2, ROUGE-L evaluation
├── app.py                      ← Flask web application
├── requirements.txt
└── README.md
```

---

## 🚀 Setup in VS Code

### Step 1 – Open folder
```
File → Open Folder → select financial_news_summarizer/
```

### Step 2 – Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 – Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 – Download NLTK data (one-time)
```bash
python -c "import nltk; [nltk.download(x) for x in ['punkt','stopwords','averaged_perceptron_tagger','punkt_tab']]"
```

### Step 5 – Run the web app
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser.

---

## 🧪 Testing

### Test from command line
```bash
python summarizer.py          # runs built-in demo article
python preprocessing.py       # tests preprocessing pipeline
python evaluate.py            # ROUGE demo on sample article
```

### Test with your dataset
```bash
# Evaluate on 20 articles from your CSV
python evaluate.py --csv dataset/financial_news.csv --col article --n 20

# Train TF-IDF vocabulary on your corpus
python model_training.py --mode tfidf --csv dataset/financial_news.csv --n 500
```

---

## 🌐 Web App Features

| Feature | Description |
|---------|-------------|
| Paste article | Multi-line input with Ctrl+Enter shortcut |
| Load samples | Two built-in demo articles (Apple, Fed) |
| Sentence count | Adjustable 1–15 sentences in summary |
| ROUGE scores | Toggle on/off ROUGE-1/2/L evaluation |
| Sentence scores | See how each sentence was ranked |
| Compression ratio | Words saved vs. original |

---

## 🧠 How It Works

```
Raw Article
    │
    ▼
Text Cleaning → remove URLs, HTML, normalize symbols
    │
    ▼
Sentence Tokenization (NLTK sent_tokenize)
    │
    ▼
Word Frequency (TF) → normalize, boost financial keywords
    │
    ▼
Sentence Scoring:
    base TF score
  + financial keyword density bonus
  + number/figure detection bonus
  + position bonus (first 20% of article)
  × length factor
    │
    ▼
MMR Re-ranking (Maximal Marginal Relevance)
  → TF-IDF cosine similarity matrix
  → balance relevance vs. diversity
    │
    ▼
Select Top-N sentences → restore original order
    │
    ▼
Summary Output
```

---

## 📊 Recommended Kaggle Datasets

| Dataset | Link | Column |
|---------|------|--------|
| All the News | https://www.kaggle.com/datasets/snapcrack/all-the-news | `content` |
| Financial News | https://www.kaggle.com/datasets/notlucasp/financial-news-headlines | `content` |
| Reuters Financial | https://www.kaggle.com/datasets/daittan/reuters-financial-articles | `article` |

Download, place CSV in `dataset/`, then:
```bash
python model_training.py --mode tfidf --csv dataset/your_file.csv --col content
```

---

## 📐 ROUGE Score Interpretation

| Score | Meaning |
|-------|---------|
| ROUGE-1 F1 > 0.5 | Good unigram coverage |
| ROUGE-2 F1 > 0.2 | Good phrase preservation |
| ROUGE-L F1 > 0.4 | Good structural similarity |

*Note: Financial extractive summarizers typically score ROUGE-1 F1 of 0.45–0.65
when tested against pseudo-references (first 30% of article).*

---

## ⚙️ Configuration

Edit at top of each file:

| File | Parameter | Default | Effect |
|------|-----------|---------|--------|
| `summarizer.py` | `num_sentences` | 5 | Summary length |
| `summarizer.py` | `use_semantic` | False | SBERT embeddings |
| `model_training.py` | `max_features` | 20000 | TF-IDF vocab size |
| `evaluate.py` | `compression_ratio` | 0.3 | Reference size |

---

## 🛠 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Re-activate venv, re-run `pip install -r requirements.txt` |
| `punkt not found` | Run the NLTK download command in Step 4 |
| Port 5000 in use | Change to `app.run(port=5001)` in app.py |
| Slow first run | NLTK downloads data once; subsequent runs are fast |
| Low ROUGE scores | Use more sentences or switch `use_semantic=True` |

---

*Built with NLTK · Scikit-learn · Flask · Python*
