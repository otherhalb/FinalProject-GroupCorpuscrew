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

#%%
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

# ===========================================================
# 1. LOAD DATA
# ===========================================================

PATH_TO_CSV = "../Data/threads_reviews.csv"
df = pd.read_csv(PATH_TO_CSV)

print("Dataset Loaded.")
print(df.head())


# ===========================================================
# 2. BASIC EDA
# ===========================================================

print("\n--- BASIC INFO ---")
print(df.info())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- RATING DISTRIBUTION ---")
if "rating" in df.columns:
    print(df["rating"].value_counts())


# ===========================================================
# 3. DUPLICATE DETECTION (RAW TEXT)
# ===========================================================

print("\n===== RAW DUPLICATE CHECK =====")

dups_raw = df[df.duplicated(subset=["review_description"], keep=False)]
dups_raw_idx = dups_raw.reset_index()[["index", "review_description"]]

print("Total RAW duplicate rows:", len(dups_raw_idx))
print("\n--- SAMPLE RAW DUPLICATES ---")
print(dups_raw_idx.head(10))


# ===========================================================
# 3A. Long-review duplicate removal
# ===========================================================

def is_long_review(text, min_words=15):
    return len(str(text).split()) >= min_words

df["is_long"] = df["review_description"].apply(is_long_review)

df_long = df[df["is_long"] == True].copy()
df_short = df[df["is_long"] == False].copy()

print("\n===== LONG REVIEW STATS =====")
print("Long reviews:", df_long.shape[0])
print("Short reviews:", df_short.shape[0])

dup_long_mask = df_long.duplicated(subset=["review_description"], keep=False)
df_long_dups = df_long[dup_long_mask]

print("\n===== LONG DUPLICATES FOUND =====")
print("Total long duplicate rows:", df_long_dups.shape[0])

dup_groups_long = (
    df_long_dups.reset_index()
                .groupby("review_description")["index"]
                .apply(list)
                .reset_index(name="row_numbers")
)

print("Distinct long duplicated texts:", len(dup_groups_long))
print("\n--- SAMPLE LONG DUPLICATE GROUPS ---")
print(dup_groups_long.head(10))

before_long = df_long.shape[0]
df_long_clean = df_long.drop_duplicates(subset=["review_description"], keep="first")
after_long = df_long_clean.shape[0]

print("\n===== LONG DUPLICATE REMOVAL =====")
print("Before:", before_long)
print("After :", after_long)
print("Removed:", before_long - after_long)

df = pd.concat([df_long_clean, df_short], ignore_index=True)
df.drop(columns=["is_long"], inplace=True)

print("\n===== AFTER LONG-DEDUPE: df.shape =====")
print(df.shape)


# ===========================================================
# 4. BRAND NORMALIZATION
# ===========================================================

brand_map = {

    # Twitter / X
    r"\btwitter\b": "twitter",
    r"\btwt\b": "twitter",
    r"\btwit\b": "twitter",
    r"\btwwitter\b": "twitter",
    r"\btwiter\b": "twitter",
    r"\bx\b": "twitter",

    # Threads
    r"\bthreads\b": "threads",
    r"\bthread\b": "threads",
    r"\btread\b": "threads",
    r"\btreads\b": "threads",
    r"\bthreds\b": "threads",

    # Instagram
    r"\binstagram\b": "instagram",
    r"\binsta\b": "instagram",
    r"\binstgram\b": "instagram",
    r"\binstragram\b": "instagram",

    # Facebook + Meta
    r"\bfacebook\b": "facebook",
    r"\bfacebok\b": "facebook",
    r"\bfb\b": "facebook",
    r"\bmeta\b": "facebook",   # <-- Meta → Facebook
}

brand_list = {"twitter", "threads", "instagram", "facebook"}


def normalize_brands(text):
    text = str(text).lower()
    for pattern, repl in brand_map.items():
        text = re.sub(pattern, repl, text)
    return text


# Apply brand normalization
df["review_norm"] = df["review_description"].apply(normalize_brands)

print("\n===== BRAND NORMALIZATION CHECK =====")
print(df[["review_description", "review_norm"]].head(10))


# ===========================================================
# 6.  CLEANING PIPELINE
# ===========================================================

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):

    # Step 1: Normalize brands
    text_norm = normalize_brands(text)

    # Step 2: Light clean
    clean = emoji.replace_emoji(text_norm, replace="")
    clean = re.sub(r"http\S+|www\S+", "", clean)
    clean = clean.lower()
    clean = re.sub(r"[^a-zA-Z0-9 ]", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    # Step 3: Token clean
    tokens = nltk.word_tokenize(clean)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(tokens)

# Apply full cleaning
df["review_cleaned"] = df["review_norm"].apply(clean_text)

print("\n===== CLEANING CHECK =====")
for i in range(5):
    print("\nRaw:", df.loc[i, "review_description"])
    print("Cleaned:", df.loc[i, "review_cleaned"])


# ===========================================================
# 7. REMOVE EMPTY CLEANED ROWS
# ===========================================================

before = df.shape[0]
df = df[df["review_cleaned"].str.strip() != ""]
after = df.shape[0]

print("\n===== EMPTY CLEANED TEXT REMOVAL =====")
print("Before:", before)
print("After:", after)
print("Removed:", before - after)


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

<<<<<<< Updated upstream
# ===========================================
# 5. 
# ===========================================
# %%
=======
>>>>>>> Stashed changes
