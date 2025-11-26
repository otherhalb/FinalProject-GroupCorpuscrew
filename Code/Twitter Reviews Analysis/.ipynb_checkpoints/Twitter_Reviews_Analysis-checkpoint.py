
## Twitter (X) Sentiment Analysis 

# This script analyzes user sentiment toward **Twitter (now X)** during the same period as Threads’ launch in **July 2023**.  
# It examines [34,788] Google Play reviews to compare public perception, usability, and satisfaction with Twitter’s platform.


## 📑 Table of Contents
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
#   - Sentiment trends and comparative insights
#9. **Conclusion**
#   - Observations and sentiment patterns for Twitter


#**Dataset:** [2 Million X (Formerly Twitter) Google Reviews — Kaggle](https://www.kaggle.com/datasets/bwandowando/2-million-formerly-twitter-google-reviews)  
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

PATH_TO_CSV = "../Data/twitter_2023-07.csv"
df_tw = pd.read_csv(PATH_TO_CSV)

print("Dataset Loaded.")
print(df_tw.head())


# ===========================================================
# 2. BASIC EDA
# ===========================================================

print("\n--- BASIC INFO ---")
print(df_tw.info())

print("\n--- MISSING VALUES ---")
print(df_tw.isnull().sum())

# Identify rows where review_text is missing or empty
mask_empty = df_tw["review_text"].isna() | (df_tw["review_text"].astype(str).str.strip() == "")
empty_rows = df_tw[mask_empty]

print("\n--- SAMPLE EMPTY REVIEW TEXT ROWS ---")
print(empty_rows)

print("\n--- RATING DISTRIBUTION ---")
if "review_rating" in df_tw.columns:
    print(df_tw["review_rating"].value_counts())

# ===========================================================
# 3. REMOVE NULL REVIEW TEXT
# ===========================================================
before = df_tw.shape[0]

df_tw["review_text"] = df_tw["review_text"].fillna("")
df_tw = df_tw[df_tw["review_text"].str.strip() != ""]

after = df_tw.shape[0]

print("\n===== REMOVAL SUMMARY (EMPTY review_text) =====")
print(f"Rows before: {before}")
print(f"Rows after : {after}")
print(f"Removed    : {before - after}")


# ===========================================================
# 4. DUPLICATE DETECTION (RAW TEXT)
# ===========================================================

print("\n===== RAW DUPLICATE CHECK =====")

dups_raw = df_tw[df_tw.duplicated(subset=["review_text"], keep=False)]
dups_raw_idx = dups_raw.reset_index()[["index", "review_text"]]

print("Total RAW duplicate rows:", len(dups_raw_idx))
print("\n--- SAMPLE RAW DUPLICATES ---")
print(dups_raw_idx.head(10))

# 4A. Long-review duplicate check
def is_long_review(text, min_words=15):
    return len(str(text).split()) >= min_words

df_tw["is_long"] = df_tw["review_text"].apply(is_long_review)

df_long = df_tw[df_tw["is_long"] == True].copy()
df_short = df_tw[df_tw["is_long"] == False].copy()

print("\n===== LONG REVIEW STATS =====")
print("Long reviews:", df_long.shape[0])
print("Short reviews:", df_short.shape[0])

dup_long_mask = df_long.duplicated(subset=["review_text"], keep=False)
df_long_dups = df_long[dup_long_mask]

print("\n===== LONG DUPLICATES FOUND =====")
print("Total long duplicate rows:", df_long_dups.shape[0])

dup_groups_long = (
    df_long_dups.reset_index()
                .groupby("review_text")["index"]
                .apply(list)
                .reset_index(name="row_numbers")
)

print("Distinct long duplicated texts:", len(dup_groups_long))

# ===========================================================
# 5. BRAND NORMALIZATION
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
    r"\bmeta\b": "facebook",
}

def normalize_brands(text):
    text = str(text).lower()
    for pattern, repl in brand_map.items():
        text = re.sub(pattern, repl, text)
    return text

df_tw["review_norm"] = df_tw["review_text"].apply(normalize_brands)

print("\n===== BRAND NORMALIZATION CHECK =====")
print(df_tw[["review_text", "review_norm"]].head(10))

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
df_tw["review_cleaned"] = df_tw["review_norm"].apply(clean_text)

print("\n===== CLEANING CHECK =====")
for i in range(5):
    print("\nRaw:", df_tw.loc[i, "review_text"])
    print("Cleaned:", df_tw.loc[i, "review_cleaned"])



# ===========================================================
# 7. REMOVE EMPTY CLEANED ROWS
# ===========================================================

before = df_tw.shape[0]
df_tw = df_tw[df_tw["review_cleaned"].str.strip() != ""]
after = df_tw.shape[0]

print("\n===== EMPTY CLEANED TEXT REMOVAL =====")
print("Before:", before)
print("After:", after)
print("Removed:", before - after)

# ===========================================
# 8. Sentiment Analysis (VADER) - using Uncleaned Original Text
# ===========================================
analyzer = SentimentIntensityAnalyzer()

df_tw["vader_score"] = df_tw["review_text"].astype(str).apply(
    lambda x: analyzer.polarity_scores(x)["compound"]
)

print("\n--- Overall VADER Sentiment Summary ---")
print(df_tw["vader_score"].describe())

def categorize_sentiment(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

df_tw["sentiment_label"] = df_tw["vader_score"].apply(categorize_sentiment)

print("\n--- Sentiment Label Distribution ---")
print(df_tw["sentiment_label"].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(data=df_tw, x="sentiment_label", palette="coolwarm")
plt.title("Sentiment Distribution (VADER)")
plt.xlabel("Sentiment Category")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

