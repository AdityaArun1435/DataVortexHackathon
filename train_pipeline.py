"""
Data Vortex Round 2 - Sentiment Classification Pipeline
Rebuilding the Social Engine's semantic comprehension layer.

Task chosen: 3-class sentiment classification (Positive / Negative / Neutral)
on Dataset 2 (recovered from the corrupted-column PDF via extract_dataset.py).

Run: python3 train_pipeline.py
"""

import re
import string
import pandas as pd
import numpy as np
import joblib

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

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------
df = pd.read_csv('recovered_dataset_raw.csv')
print(f"Loaded {len(df)} labeled posts.")
print(df['sentiment_label'].value_counts())

# ---------------------------------------------------------------------------
# 2. PREPROCESSING
# ---------------------------------------------------------------------------
stop_words = set(stopwords.words('english'))
# keep negation words - critical for sentiment, stripping them flips meaning
NEGATIONS = {'no', 'not', 'nor', "don't", "didn't", "doesn't", "isn't",
             "wasn't", "won't", "can't", "couldn't", "shouldn't", "wouldn't"}
stop_words = stop_words - NEGATIONS

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'@user', ' ', text)              # anonymized mentions carry no sentiment signal
    text = re.sub(r'http\S+|www\S+', ' ', text)      # URLs
    text = re.sub(r'#(\w+)', r'\1', text)            # keep hashtag word, drop the '#'
    text = re.sub(r'\\u[0-9a-fA-F]{4}', "'", text)   # stray unicode escape artifacts (e.g. \u2019)
    text = re.sub(r'[0-9]+', ' ', text)              # numbers rarely carry sentiment here
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    return ' '.join(tokens)

df['clean_text'] = df['post_text'].apply(clean_text)
df = df[df['clean_text'].str.len() > 0].reset_index(drop=True)
print(f"\nAfter cleaning: {len(df)} rows retained.")
print(df[['post_text', 'clean_text']].head(5).to_string())

# ---------------------------------------------------------------------------
# 3. TRAIN/TEST SPLIT
# ---------------------------------------------------------------------------
X = df['clean_text']
y = df['sentiment_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# ---------------------------------------------------------------------------
# 4. FEATURE EXTRACTION
# ---------------------------------------------------------------------------
vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),   # unigrams + bigrams (catches "not good" vs "good")
    min_df=3,
    sublinear_tf=True
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")

# ---------------------------------------------------------------------------
# 5. MODEL COMPARISON
# ---------------------------------------------------------------------------
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
    print(f"\n{name}: accuracy={acc:.4f}, macro-F1={f1_macro:.4f}")

best_name = max(results, key=lambda k: results[k]['f1_macro'])
best = results[best_name]
print(f"\n>>> Best model: {best_name} (macro-F1={best['f1_macro']:.4f})")

# ---------------------------------------------------------------------------
# 6. DETAILED EVALUATION OF BEST MODEL
# ---------------------------------------------------------------------------
report = classification_report(y_test, best['preds'], digits=3)
print("\n" + report)

cm = confusion_matrix(y_test, best['preds'], labels=['Positive', 'Negative', 'Neutral'])
print("Confusion matrix (rows=true, cols=pred), labels=[Positive, Negative, Neutral]:")
print(cm)

# ---------------------------------------------------------------------------
# 7. ERROR ANALYSIS - save misclassified examples
# ---------------------------------------------------------------------------
test_df = pd.DataFrame({
    'text': X_test.values,
    'true_label': y_test.values,
    'pred_label': best['preds']
})
errors = test_df[test_df['true_label'] != test_df['pred_label']]
errors.to_csv('misclassified_examples.csv', index=False)
print(f"\n{len(errors)} misclassified examples saved out of {len(test_df)} test rows "
      f"({len(errors)/len(test_df)*100:.1f}% error rate).")

# ---------------------------------------------------------------------------
# 8. SAVE ARTIFACTS
# ---------------------------------------------------------------------------
joblib.dump(best['model'], 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

with open('model_comparison.txt', 'w') as f:
    for name, r in results.items():
        f.write(f"{name}: accuracy={r['accuracy']:.4f}, macro-F1={r['f1_macro']:.4f}\n")
    f.write(f"\nBest model: {best_name}\n\n")
    f.write(report)
    f.write(f"\nConfusion matrix (rows=true, cols=pred), labels=[Positive, Negative, Neutral]:\n{cm}\n")

print("\nSaved: sentiment_model.pkl, tfidf_vectorizer.pkl, model_comparison.txt, misclassified_examples.csv")
