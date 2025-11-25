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

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
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

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"http\S+|www\S+", "", text) # removes url
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)

# Apply cleaning
df["review_cleaned"] = df["review_description"].apply(clean_text)

print("\n--- CLEANING CHECK: original vs cleaned ---")
for i in range(3):
    print("\nOriginal:", df.loc[i, "review_description"])
    print("Cleaned :", df.loc[i, "review_cleaned"])

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
plt.xlabel("Sentiment Category")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ===========================================
# 5. 
# ===========================================