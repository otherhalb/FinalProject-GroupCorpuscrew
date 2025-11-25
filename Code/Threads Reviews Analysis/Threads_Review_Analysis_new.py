# Threads Sentiment Analysis — NLP Comparison Project

#This script analyzes user sentiment toward **Threads (Instagram App)** using Natural Language Processing (NLP) techniques.  
#It processes 32,910 user reviews collected in **July 2023**, coinciding with Threads’ public launch, to study platform reception and user opinions.


## Table of Contents
#1. **Import Libraries**
#2. **Load Dataset**
#3. **Exploratory Data Analysis (EDA)**
#4. **Data Preprocessing**
#   - Lowercasing and punctuation removal  
#   - Emoji and stop-word filtering  
#   - Lemmatization and tokenization
#5. **Feature Extraction**
#   - Bag of Words / TF-IDF representation
#6. **Model Training**
#   - Logistic Regression (Baseline)
#   - Deep Learning Models (e.g., LSTM/BERT)
#7. **Model Evaluation**
#   - Accuracy, Precision, Recall, F1-score
#8. **Visualization**
#   - Sentiment distribution and keyword analysis
#9. **Conclusion**
#   - Observations and sentiment trends for Threads

#**Dataset:** [Threads: An Instagram App Reviews — Kaggle](https://www.kaggle.com/datasets/saloni1712/threads-an-instagram-app-reviews/data)  
#**Institution:** George Washington University · Fall 2025

# ===========================================
# Load Libraries
# ===========================================
# install if necessary:
# pip install langdetect

import pandas as pd
import numpy as np
import re
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# BERTopic
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Accurate English detection
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 42  # reproducible language detection

# Reproducibility
import random
import torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Download NLTK dependencies
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("vader_lexicon")

# ===========================================
# 1. Load Dataset
# ===========================================
PATH_TO_CSV = "../Data/threads_reviews.csv"   # <--- change this to your actual path
df = pd.read_csv(PATH_TO_CSV)

print("Dataset Loaded.")
print(df.head())

# ===========================================
# 2. Basic EDA
# ===========================================
print("\n--- BASIC INFO ---")
print(df.info())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- RATING DISTRIBUTION ---")
if "rating" in df.columns:
    print(df["rating"].value_counts())

# ===========================================
# 3. Data Preprocessing
#    (Keep original text, create new cleaned column)
# ===========================================
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# English Detection
def is_english(text):
    try:
        lang = detect(text)
        return lang == "en"
    except:
        return False
    
def clean_text(text):
    text = str(text).strip()
    text = re.sub(r"http\S+|www\S+", "", text) # Remove URLs
    text = emoji.replace_emoji(text, replace="")
    text = text.lower()
    # English filter
    if not is_english(text):
        return ""

    text = re.sub(r"(.)\1{3,}", r"\1\1", text) # Remove garbage (long repeated chars)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    # If empty after cleaning → remove
    if len(tokens) < 3:
        return ""

    return " ".join(tokens)

# Apply cleaning
df["review_cleaned"] = df["review_description"].apply(clean_text)

print("\n--- CLEANING CHECK: original vs cleaned ---")
for i in range(3):
    print("\nOriginal:", df.loc[i, "review_description"])
    print("Cleaned :", df.loc[i, "review_cleaned"])

# Drop empty cleaned rows
df = df[df["review_cleaned"] != ""].copy()

# ===========================================
# 4. Sentiment Analysis (VADER) - using Uncleaned Original Text
# ===========================================
# Initialize analyzer
analyzer = SentimentIntensityAnalyzer()

# Compute compound sentiment score using ORIGINAL text (not cleaned)
df["vader_score"] = df["review_description"].astype(str).apply(
    lambda x: analyzer.polarity_scores(x)["compound"]
)

print("\n--- Overall VADER Sentiment Summary ---")
print(df["vader_score"].describe())

# Sentiment Category Labeling
def categorize_sentiment(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

df["sentiment_label"] = df["vader_score"].apply(categorize_sentiment)

print("\n--- Sentiment Label Distribution ---")
print(df["sentiment_label"].value_counts())

# Visualization: Sentiment Distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="sentiment_label", palette="coolwarm")
plt.title("Sentiment Distribution (VADER)")
plt.tight_layout()
plt.show()

# ===========================================
# 5. Sentiment Grouping: LDA, NMF, BERTopic
# ===========================================
# LDA

# NMF

# BERTopic
# Split into 3 sentiment datasets
positive_df = df[df["sentiment_label"] == "positive"].copy()
neutral_df  = df[df["sentiment_label"] == "neutral"].copy()
negative_df = df[df["sentiment_label"] == "negative"].copy()

positive_texts = positive_df["review_cleaned"].tolist()
neutral_texts  = neutral_df["review_cleaned"].tolist()
negative_texts = negative_df["review_cleaned"].tolist()

print("Counts:", len(positive_texts), len(neutral_texts), len(negative_texts))

# BERTopic for Each Sentiment (UMAP Disabled)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def run_bertopic(texts, min_topic_size=20):
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    model = BERTopic(
        language="english",
        umap_model=None,
        min_topic_size=min_topic_size,
        calculate_probabilities=True,
        verbose=True
    )
    topics, probs = model.fit_transform(texts, embeddings)
    return model, topics, probs, embeddings


print("\nRunning BERTopic for POSITIVE reviews...")
pos_model, pos_topics, pos_probs, pos_emb = run_bertopic(positive_texts)

print("\nRunning BERTopic for NEUTRAL reviews...")
neu_model, neu_topics, neu_probs, neu_emb = run_bertopic(neutral_texts)

print("\nRunning BERTopic for NEGATIVE reviews...")
neg_model, neg_topics, neg_probs, neg_emb = run_bertopic(negative_texts)

# Reduce to 5 Topics Each
def reduce_to_5(model, texts, embeddings):
    reduced_model = model.reduce_topics(texts, nr_topics=5)
    new_topics, new_probs = reduced_model.transform(texts, embeddings=embeddings)
    return reduced_model, new_topics, new_probs

pos_model5, pos_topics5, pos_probs5 = reduce_to_5(pos_model, positive_texts, pos_emb)
neu_model5, neu_topics5, neu_probs5 = reduce_to_5(neu_model, neutral_texts, neu_emb)
neg_model5, neg_topics5, neg_probs5 = reduce_to_5(neg_model, negative_texts, neg_emb)

positive_df["topic5"] = pos_topics5
neutral_df["topic5"]  = neu_topics5
negative_df["topic5"] = neg_topics5

# Diagnostics: Topic Tables, Top Words, Barplots, Docs
def inspect_topics(model, df_sent, group_name):
    print("\n" + "="*80)
    print(f"===== BERTOPIC SUMMARY ({group_name.upper()}) — 5 TOPICS =====")
    print("="*80)

    topic_info = model.get_topic_info()
    print(topic_info)

    # Top words
    print(f"\n--- TOP WORDS ({group_name}) ---")
    for tid in topic_info["Topic"]:
        print(f"\nTopic {tid}:")
        print(model.get_topic(tid))

    # Barplot
    plt.figure(figsize=(8, 4))
    sns.barplot(x="Topic", y="Count", data=topic_info, palette="viridis")
    plt.title(f"Topic Distribution ({group_name.capitalize()} Reviews)")
    plt.tight_layout()
    plt.show()

    # Representative docs
    print(f"\n--- REPRESENTATIVE DOCS ({group_name}) ---")
    for tid in topic_info["Topic"]:
        docs = model.get_representative_docs(tid)  # returns a LIST
        print(f"\nTopic {tid} Examples:")
        for d in docs[:3]:
            print("-", d[:200], "...")


inspect_topics(pos_model5, positive_df, "positive")
inspect_topics(neu_model5, neutral_df, "neutral")
inspect_topics(neg_model5, negative_df, "negative")

