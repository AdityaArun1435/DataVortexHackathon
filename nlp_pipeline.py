"""
Data Vortex Round 2 - End-to-End NLP Pipeline
Rebuilding the Social Engine's semantic comprehension layer.

Combines dataset recovery (from the obfuscated PDF) and the full sentiment
classification pipeline (preprocessing, TF-IDF, model comparison, training,
evaluation) into one script.

Run: python3 nlp_pipeline.py
Requires: pandas, scikit-learn, nltk, pdfplumber, joblib, matplotlib
"""

import re
import string
import glob
import pandas as pd
import numpy as np
import joblib
import pdfplumber

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RANDOM_STATE = 42
PDF_PATH = 'Labeled_Social_NLP_Training_Data.pdf'


# ===========================================================================
# PART 1 — DATASET RECOVERY
# ===========================================================================
# The provided PDF renders each column (text_id, post_text, sentiment_label,
# topic_category) as a SEPARATE text object anchored at a fixed x-position on
# each row. Because post_text is long and overflows rightward, it visually
# overlaps the sentiment_label/topic_category columns, and any naive text
# extractor (or copy-paste) that reads left-to-right by x-position interleaves
# the characters into garbage. The underlying character stream is not
# actually corrupted, the columns just need to be re-separated by their known
# fixed x-anchors instead of by reading order.

ANCHORS = [
    ('text_id', 52.65),
    ('post_text', 128.4),
    ('sentiment_label', 204.15),
    ('topic_category', 279.9),
]
TOL = 1.5


def recover_dataset(pdf_path):
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            chars = page.chars
            if not chars:
                continue
            lines = {}
            for c in chars:
                key = round(c['top'], 0)
                lines.setdefault(key, []).append(c)

            for top, line_chars in sorted(lines.items()):
                if page_num == 0 and top < 60:
                    continue  # header row

                col_idx = 0
                col_text = {name: [] for name, _ in ANCHORS}
                last_x = None
                for c in line_chars:
                    x0 = c['x0']
                    if col_idx == 0 and x0 >= ANCHORS[1][1] - TOL:
                        col_idx = 1
                    elif col_idx == 1 and last_x is not None and x0 < last_x - 3:
                        col_idx = 2  # post_text ended, sentiment_label starts
                    elif col_idx == 2 and last_x is not None and x0 < last_x - 3:
                        col_idx = 3  # sentiment_label ended, topic_category starts
                    elif col_idx == 2 and x0 >= ANCHORS[3][1] - TOL:
                        col_idx = 3  # short label, topic_category starts at its own anchor
                    col_name = ANCHORS[col_idx][0]
                    col_text[col_name].append(c['text'])
                    last_x = x0

                row = {name: ''.join(col_text[name]).strip() for name, _ in ANCHORS}
                if row['text_id']:
                    rows.append(row)

    return pd.DataFrame(rows)


print("=== PART 1: Recovering dataset from obfuscated PDF ===")
df = recover_dataset(PDF_PATH)
print(f"Recovered {len(df)} rows.")
print(df['sentiment_label'].value_counts())
df.to_csv('recovered_dataset_raw.csv', index=False)


# ===========================================================================
# PART 2 — PREPROCESSING
# ===========================================================================
print("\n=== PART 2: Preprocessing ===")

stop_words = set(stopwords.words('english'))
NEGATIONS = {'no', 'not', 'nor', "don't", "didn't", "doesn't", "isn't",
             "wasn't", "won't", "can't", "couldn't", "shouldn't", "wouldn't"}
stop_words = stop_words - NEGATIONS


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'@user', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'\\u[0-9a-fA-F]{4}', "'", text)
    text = re.sub(r'[0-9]+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    return ' '.join(tokens)


df['clean_text'] = df['post_text'].apply(clean_text)
df = df[df['clean_text'].str.len() > 0].reset_index(drop=True)
print(f"After cleaning: {len(df)} rows retained.")


# ===========================================================================
# PART 3 — TRAIN/TEST SPLIT + FEATURES
# ===========================================================================
print("\n=== PART 3: Split + TF-IDF ===")

X = df['clean_text']
y = df['sentiment_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

vectorizer = TfidfVectorizer(
    max_features=8000, ngram_range=(1, 2), min_df=3, sublinear_tf=True
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")


# ===========================================================================
# PART 4 — MODEL COMPARISON & TRAINING
# ===========================================================================
print("\n=== PART 4: Model comparison ===")

models = {
    'Multinomial Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE),
    'Linear SVM': LinearSVC(C=1.0, random_state=RANDOM_STATE, max_iter=5000),
}

results = {}
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)
    _, _, f1_macro, _ = precision_recall_fscore_support(y_test, preds, average='macro')
    results[name] = {'model': model, 'preds': preds, 'accuracy': acc, 'f1_macro': f1_macro}
    print(f"{name}: accuracy={acc:.4f}, macro-F1={f1_macro:.4f}")

best_name = max(results, key=lambda k: results[k]['f1_macro'])
best = results[best_name]
print(f"\n>>> Best model: {best_name} (macro-F1={best['f1_macro']:.4f})")


# ===========================================================================
# PART 5 — DETAILED EVALUATION
# ===========================================================================
print("\n=== PART 5: Evaluation ===")

report = classification_report(y_test, best['preds'], digits=3)
print(report)

cm = confusion_matrix(y_test, best['preds'], labels=['Positive', 'Negative', 'Neutral'])
print("Confusion matrix (rows=true, cols=pred), labels=[Positive, Negative, Neutral]:")
print(cm)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(cm, cmap='Blues')
labels = ['Positive', 'Negative', 'Neutral']
ax.set_xticks(range(3)); ax.set_xticklabels(labels)
ax.set_yticks(range(3)); ax.set_yticklabels(labels)
ax.set_xlabel('Predicted label', fontweight='bold')
ax.set_ylabel('True label', fontweight='bold')
ax.set_title(f'Confusion Matrix — {best_name}', fontweight='bold')
thresh = cm.max() / 2
for i in range(3):
    for j in range(3):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black', fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=160)
print("Saved confusion_matrix.png")


# ===========================================================================
# PART 6 — ERROR ANALYSIS
# ===========================================================================
print("\n=== PART 6: Error analysis ===")

test_df = pd.DataFrame({
    'text': X_test.values,
    'true_label': y_test.values,
    'pred_label': best['preds']
})
errors = test_df[test_df['true_label'] != test_df['pred_label']]
errors.to_csv('misclassified_examples.csv', index=False)
print(f"{len(errors)} misclassified out of {len(test_df)} "
      f"({len(errors)/len(test_df)*100:.1f}% error rate). Saved misclassified_examples.csv")


# ===========================================================================
# PART 7 — SAVE ARTIFACTS
# ===========================================================================
print("\n=== PART 7: Saving artifacts ===")

# bundle model + vectorizer into a single file for submission
joblib.dump({'model': best['model'], 'vectorizer': vectorizer}, 'sentiment_model_bundle.pkl')

with open('model_comparison.txt', 'w') as f:
    for name, r in results.items():
        f.write(f"{name}: accuracy={r['accuracy']:.4f}, macro-F1={r['f1_macro']:.4f}\n")
    f.write(f"\nBest model: {best_name}\n\n{report}\n")
    f.write(f"Confusion matrix (rows=true, cols=pred), labels=[Positive, Negative, Neutral]:\n{cm}\n")

print("Saved: recovered_dataset_raw.csv, sentiment_model_bundle.pkl, "
      "confusion_matrix.png, misclassified_examples.csv, model_comparison.txt")
print("\nDone.")
