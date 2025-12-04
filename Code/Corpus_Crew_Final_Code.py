# =============================================================
# Comparative Analysis: Threads and Twitter Reviews
# =============================================================
# import libraries
import pandas as pd
import numpy as np
import emoji
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import NMF
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
from gensim.utils import simple_preprocess
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from sklearn.linear_model import LogisticRegression
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.probability import FreqDist
from scipy.stats import pearsonr, spearmanr
# For BERTopic
import os
import random
import numpy as np
import torch
# For Langdetect
from langdetect import detect, LangDetectException, DetectorFactory

DetectorFactory.seed = 42

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HDBSCAN_RANDOM_STATE"] = "42"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from hdbscan import HDBSCAN
from umap import UMAP

umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    random_state=SEED
)

hdbscan_model = HDBSCAN(
    min_cluster_size=20,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)

# Make sure you have NLTK punkt tokenizer
nltk.download("punkt")

# =============================================================
# 1. Load the datasets and check data
# The original data did not include the sentiment_true column. 
# We added it manually for about 3,000 rows to compare human labels with VADER labels.
# =============================================================

threads_df = pd.read_csv("Data/threads_reviews_labelled.csv")     
twitter_df = pd.read_csv("Data/twitter_reviews_labelled.csv") 

### Looking at Threads dataset
# First 5 Threads
print(threads_df.head())

# Last 5 threads
print(threads_df.tail())

# count number by source
threads_df["source"].value_counts()

# ------------------------------------------------------------
# threads dataset has few 'app_store' data points.
# There are 2640 rows for data from app_store.
# ------------------------------------------------------------

### Looking at Twitter dataset
# First 5 Twitter
print(twitter_df.head())

# Last 5 twitter
print(twitter_df.tail())

# ------------------------------------------------------------
# We detect different language other than english, we will handle this in data preprocessing.
# ------------------------------------------------------------

### Threads Info
# Info print for threads
print(threads_df.info(), "\n")

### Twitter Info
# Info print for twitter
print(twitter_df.info())

# Columns for confirmation
print("Threads columns:", threads_df.columns.tolist())
print("Twitter columns:", twitter_df.columns.tolist())

# ------------------------------------------------------------
# Threads data and its review_description can be compared to Twitter data’s review_text.
# We see Twitter has unwanted columns not required for our project
# ------------------------------------------------------------

### Missing values - Threads
# Missing Values of Threads
print("\nMissing values (Threads):\n", threads_df.isnull().sum())

### Missing Values - Twitter
# Missing Values of Twitter
print("\nMissing values (Twitter):\n", twitter_df.isnull().sum())

# ------------------------------------------------------------
# Twitter has missing values in 2 columns 
# ------------------------------------------------------------


# =============================================================
# 2. Data Cleaning Pipeline
# We used slightly different pipeline for each data.
#
# For Threads:
# Removed duplicated content (raw + long-review duplicates)
# Standardized brand names
# Removed noise, emojis, special chars, stopwords
# Handled negations properly
# Removed empty texts after cleaning
# Detected language and removed unwanted languages
#
# For Twitter:
# Applied different custom stopwords, language filtering rules (Twitter tends to have much more global usage → more foreign languages)
# Everything else (duplicate logic, text cleaning, brand normalization workflow) is the same.
# =============================================================

### Data Cleaning Pipeline For Threads
# ------------------------------------------------------------
# Part 1. DUPLICATE DETECTION (RAW TEXT)
# ------------------------------------------------------------

print("\n===== RAW DUPLICATE CHECK =====")
df = threads_df.copy()
dups_raw = df[df.duplicated(subset=["review_description"], keep=False)]
dups_raw_idx = dups_raw.reset_index()[["index", "review_description"]]

print("Total RAW duplicate rows:", len(dups_raw_idx))
print("\n--- SAMPLE RAW DUPLICATES ---")
print(dups_raw_idx.head(10))


# ------------------------------------------------------------
# Part 2. LONG-REVIEW DUPLICATE REMOVAL
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Part 3. BRAND NORMALIZATION MAP
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Part 4. CLEANING PIPELINE (COMBINED)
# ------------------------------------------------------------

stop_words = set(stopwords.words("english"))
negations = {"no", "not", "nor", "dont", "can't", "cannot", "never"}
stop_words = stop_words - negations

lemmatizer = WordNetLemmatizer()

# Custom stopwords - INCLUDING normalized brand names
custom_stopwords = set([
    "app", "apps", "application", "applications",
    "threads", "experience", "im", "account"
])

def clean_text(text):
    # Step 1: Normalize brands FIRST (before any other processing)
    text = normalize_brands(text)
    
    # Step 2: Basic cleaning
    text = str(text)
    text = text.lower()
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)  # Remove special chars (added space)
    text = re.sub(r"\s+", " ", text).strip()    # Normalize whitespace
    
    # Step 3: Tokenization and stopword removal
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if word not in custom_stopwords]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)


# Apply cleaning
df["review_cleaned"] = df["review_description"].apply(clean_text)


print("\n===== CLEANING CHECK =====")
for i in range(5):
    print("\nOriginal:", df.loc[i, "review_description"])
    print("Cleaned :", df.loc[i, "review_cleaned"])

# ------------------------------------------------------------
# Part 5. REMOVE EMPTY CLEANED ROWS
# ------------------------------------------------------------

before = df.shape[0]
df = df[df["review_cleaned"].str.strip() != ""]
after = df.shape[0]

print("\n===== EMPTY CLEANED TEXT REMOVAL =====")
print("Before:", before)
print("After:", after)
print("Removed:", before - after)


# ------------------------------------------------------------
# Part 5.1 MINIMAL LANGUAGE DETECTION & FILTERING
# ------------------------------------------------------------

REMOVE_LANGS = {"vi","tr","sw","sq","so","pt","lv","lt","id","hr","et","es"}

def detect_language(text):
    try:
        text_str = str(text).strip()
        if len(text_str) < 10:
            return 'unknown'
        return detect(text_str)
    except LangDetectException:
        return 'unknown'
    except:
        return 'unknown'

# Detect languages
df["detected_lang"] = df["review_cleaned"].apply(detect_language)

print("\n===== LANGUAGE DISTRIBUTION =====")
print(df["detected_lang"].value_counts())


# REMOVE ONLY SPECIFIED LANGUAGES
# keep English + unknown + any other safe languages

before = df.shape[0]

df = df[~df["detected_lang"].isin(REMOVE_LANGS)].copy()
df = df.reset_index(drop=True)

print(f"\nRemoved {before - df.shape[0]} rows from unwanted languages.")
print("\n===== FINAL DATASET SHAPE =====")
print(df.shape)

# ------------------------------------------------------------
# Original rows ≈ 33,000
# Final rows = 29,646
# Total removed ≈ 3,300 rows (≈10%)
# ------------------------------------------------------------

### Apply cleaning to Threads Data
# Apply cleaning
threads_clean  =  df.copy()
threads_clean["review_cleaned"] = threads_clean["review_description"].apply(clean_text)


print("\n--- CLEANING CHECK: original vs cleaned ---")
for i in range(3):
    print("\nOriginal:", threads_clean.loc[i, "review_description"])
    print("Cleaned :", threads_clean.loc[i, "review_cleaned"])


### Data Cleaning Pipeline For Twitter
# ------------------------------------------------------------
# Part 1. DUPLICATE DETECTION (RAW TEXT)
# ------------------------------------------------------------

print("\n===== RAW DUPLICATE CHECK =====")
df = twitter_df.copy()
dups_raw = df[df.duplicated(subset=["review_text"], keep=False)]
dups_raw_idx = dups_raw.reset_index()[["index", "review_text"]]

print("Total RAW duplicate rows:", len(dups_raw_idx))
print("\n--- SAMPLE RAW DUPLICATES ---")
print(dups_raw_idx.head(10))


# ------------------------------------------------------------
# 2. LONG-REVIEW DUPLICATE REMOVAL
# ------------------------------------------------------------

def is_long_review(text, min_words=15):
    return len(str(text).split()) >= min_words

df["is_long"] = df["review_text"].apply(is_long_review)

df_long = df[df["is_long"] == True].copy()
df_short = df[df["is_long"] == False].copy()

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
print("\n--- SAMPLE LONG DUPLICATE GROUPS ---")
print(dup_groups_long.head(10))

before_long = df_long.shape[0]
df_long_clean = df_long.drop_duplicates(subset=["review_text"], keep="first")
after_long = df_long_clean.shape[0]

print("\n===== LONG DUPLICATE REMOVAL =====")
print("Before:", before_long)
print("After :", after_long)
print("Removed:", before_long - after_long)

df = pd.concat([df_long_clean, df_short], ignore_index=True)
df.drop(columns=["is_long"], inplace=True)

print("\n===== AFTER LONG-DEDUPE: df.shape =====")
print(df.shape)


# ------------------------------------------------------------
# 3. BRAND NORMALIZATION MAP
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 4. CLEANING PIPELINE (COMBINED)
# ------------------------------------------------------------

stop_words = set(stopwords.words("english"))
negations = {"no", "not", "nor", "dont", "can't", "cannot", "never"}
stop_words = stop_words - negations

lemmatizer = WordNetLemmatizer()

# Custom stopwords - INCLUDING normalized brand names
custom_stopwords = set([
    "app", "apps", "application", "applications",
    "twitter", "im", "account"
])

def clean_text(text):
    # Step 1: Normalize brands FIRST (before any other processing)
    text = normalize_brands(text)
    
    # Step 2: Basic cleaning
    text = str(text)
    text = text.lower()
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)  # Remove special chars (added space)
    text = re.sub(r"\s+", " ", text).strip()    # Normalize whitespace
    
    # Step 3: Tokenization and stopword removal
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if word not in custom_stopwords]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

# Apply cleaning
df["review_cleaned"] = df["review_text"].apply(clean_text)

print("\n===== CLEANING CHECK =====")
for i in range(5):
    print("\nOriginal:", df.loc[i, "review_text"])
    print("Cleaned :", df.loc[i, "review_cleaned"])


# ------------------------------------------------------------
# 5. REMOVE EMPTY CLEANED ROWS
# ------------------------------------------------------------

before = df.shape[0]
df = df[df["review_cleaned"].str.strip() != ""]
after = df.shape[0]

print("\n===== EMPTY CLEANED TEXT REMOVAL =====")
print("Before:", before)
print("After:", after)
print("Removed:", before - after)

# ------------------------------------------------------------
# Part 5.1 MINIMAL LANGUAGE DETECTION & FILTERING
# ------------------------------------------------------------

REMOVE_LANGS = {"vi","tr","tl","sw","sq","so","sl","pl","pt","lv","lt","id","hr","es","ca","cy","de","fi","hu"}

def detect_language(text):
    try:
        text_str = str(text).strip()
        if len(text_str) < 10:
            return 'unknown'
        return detect(text_str)
    except LangDetectException:
        return 'unknown'
    except:
        return 'unknown'

# Detect languages
df["detected_lang"] = df["review_cleaned"].apply(detect_language)

print("\n===== LANGUAGE DISTRIBUTION =====")
print(df["detected_lang"].value_counts())

# -----------------------------------------------------
# REMOVE ONLY SPECIFIED LANGUAGES
# keep English + unknown + any other safe languages
# -----------------------------------------------------
before = df.shape[0]

df = df[~df["detected_lang"].isin(REMOVE_LANGS)].copy()
df = df.reset_index(drop=True)

print(f"\nRemoved {before - df.shape[0]} rows from unwanted languages.")
print("\n===== FINAL DATASET SHAPE =====")
print(df.shape)

# -----------------------------------------------------
# Original rows ≈ 34,788
# Final rows = 29,611
# Total removed ≈ 5,177 rows (≈14.9%)
# -----------------------------------------------------

### Apply cleaning for Twitter
# Apply cleaning
twitter_clean = df.copy()

print("\n--- CLEANING CHECK: original vs cleaned ---")
for i in range(3):
    print("\nOriginal:", twitter_clean.loc[i, "review_text"])
    print("Cleaned :", twitter_clean.loc[i, "review_cleaned"])


# -----------------------------------------------------
# [Additional Insights] Major Insights from Top-20 Words: Threads vs Twitter
#
# - Threads shows early-stage excitement + requests:
#   "Threads is promising and feels good to use. But to compete with Twitter and Instagram, it needs more features”
# - Twitter shows dissatisfaction tied to leadership:
#   “Elon/Musk made updates that made things worse
# -----------------------------------------------------
### Threads frequent Words

# Combine all reviews (or you can do per cluster)
all_words = " ".join(threads_clean["review_cleaned"]).split()  # or nltk.word_tokenize(text)
freq_dist = FreqDist(all_words)

# Top 20 words overall
print("Top 20 words overall:")
for word, freq in freq_dist.most_common(20):
    print(f"{word}: {freq}")

# -----------------------------------------------------
# - twitter (5806), instagram (3644), facebook (1135) → heavy cross-platform comparison
# - good, nice, great, better → positive onboarding sentiment
# - follow, people, see, post → emphasis on social interaction and discovering content
# - feature, please, need, want → strong demand for missing functionality
# - use, new → early adoption and first-time user experience
#
# Threads comments show:
# - Users are actively comparing Threads with Twitter, Instagram, and Facebook.
# - The overall tone is positive, with many users praising the app’s feel and design.
# - Feedback is constructive and improvement-focused, with repeated requests for features.
# - Early users want Threads to expand functionality to become competitive with Twitter/Instagram.
# -----------------------------------------------------
### Twitter Frequent words

# Combine all reviews (or you can do per cluster)
all_words = " ".join(twitter_clean["review_cleaned"]).split()  # or nltk.word_tokenize(text)
freq_dist = FreqDist(all_words)

# Top 20 words overall
print("Top 20 words overall:")
for word, freq in freq_dist.most_common(20):
    print(f"{word}: {freq}")

# -----------------------------------------------------
# - elon (3885), musk (1907) → extremely Musk-centric discussion
# - tweet, name, change, update → focused on platform changes under Musk
# - not, no, worse, even → negative sentiment
# - get, back, since → reactions to changes and reversions
# - social → identity as a social platform under debate
#
#Twitter comments show:
# - The platform identity is tightly tied to Elon Musk.
# - Users comment heavily on updates, renaming (Twitter → X), UI changes, etc.
# - Much more complaining, negativity
# -----------------------------------------------------

# ------------------------------------------------------
## SENTIMENT ANALYSIS : USING VADER + KNN FOR THREADS
# ------------------------------------------------------
# By applying KMeans clustering to the compound sentiment scores, we could segment reviews into three clusters. The centroids of these clusters inform the thresholds for classifying sentiment, allowing for more objective labeling than manual cutoff values.

# ----------- LOAD DATA --------------
df_threads = threads_clean.copy()     
# ------------------------------------


# ---------- VADER SCORES -----------
analyzer = SentimentIntensityAnalyzer()
df_threads["compound"] = df_threads["review_cleaned"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
compound_array = df_threads["compound"].values.reshape(-1,1)
# ------------------------------------

# ----------- KMEANS 3 CLUSTERS -----
kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
clusters = kmeans.fit_predict(compound_array)
df_threads["cluster"] = clusters

# map centroids -> pos/neu/neg
centroids = kmeans.cluster_centers_.flatten()
order = np.argsort(centroids)
label_map = {order[0]: "negative", order[1]: "neutral", order[2]: "positive"}

df_threads["sentiment"] = df_threads["cluster"].map(label_map)
# ------------------------------------

print(df_threads[["review_description","review_cleaned","compound","sentiment"]].head())

# ----------- SENTIMENT DISTRIBUTION PLOT ---------
import matplotlib.pyplot as plt

df_threads["sentiment"].value_counts().plot(kind="bar")

plt.title("Sentiment Distribution (KMeans + VADER)")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.show()

# ------ BOXPLOT OF COMPOUND SCORES BY SENTIMENT -------
df_threads.boxplot(column="compound", by="sentiment", grid=False)

plt.title("Compound Score by Sentiment Cluster")
plt.suptitle("")
plt.xlabel("Sentiment")
plt.ylabel("Compound Score")
plt.tight_layout()
plt.show()


# ----------- SCATTER PLOT OF CLUSTERS -----------
# Set style
sns.set(style="whitegrid")

# Plot the clusters along the compound score
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=range(len(df_threads)), 
    y="compound", 
    hue="sentiment", 
    palette={"negative":"red", "neutral":"gray", "positive":"green"},
    data=df_threads,
    s=50
)
plt.title("KMeans Clusters of Reviews (VADER Compound Scores)")
plt.xlabel("Review Index")
plt.ylabel("Compound Score")
plt.legend(title="Sentiment")
plt.show()

# The KNN clustering makes a custom threshold for us without us needing to have a hard coded threshold for different sentiments.The reviews seem to be perfectly clustered with respect to their respective scores.

# --------------------------------------------------------------------
# ----------- WORDCLOUDS + TOP WORDS PER SENTIMENT CLUSTER -----------
# --------------------------------------------------------------------
# Define a function to get top words
def get_top_words(text_series, n=10):
    words = " ".join(text_series).split()
    counter = Counter(words)
    return counter.most_common(n)

# Plot word clouds for each sentiment cluster
sentiments = df_threads["sentiment"].unique()
plt.figure(figsize=(18,6))

for i, sentiment in enumerate(sentiments, 1):
    text = df_threads[df_threads["sentiment"] == sentiment]["review_cleaned"]
    
    # Generate word cloud
    wordcloud = WordCloud(width=400, height=300, background_color="white").generate(" ".join(text))
    
    # Plot
    plt.subplot(1, len(sentiments), i)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"{sentiment.capitalize()} Reviews")
    
    # Print top 10 words in console
    print(f"Top words in {sentiment} cluster:", get_top_words(text))

plt.tight_layout()
plt.show()

threads_pos = df_threads[df_threads["sentiment"] == "positive"]
threads_neu = df_threads[df_threads["sentiment"] == "neutral"]
threads_neg = df_threads[df_threads["sentiment"] == "negative"]

# - The negative word cloud shows frustrations about posting, missing features, and comparisons where Threads feels worse than Twitter/Instagram.
# - The positive word cloud highlights enthusiasm, with words like good, nice, great, and love showing strong early satisfaction.
# - The neutral word cloud reflects general observations about usage: copying, posting, following, and feature needs without strong sentiment.

# ------------------------------------------------------------
# ----------- STANDARDIZE MANUAL LABELS -----------
# ------------------------------------------------------------
# Clean label mapping
label_mapping = {
    'negative': 'negative',
    'positive': 'positive',
    'neutral': 'neutral',
    'Negative': 'negative',
    'Positive': 'positive',
    'Neutral': 'neutral',
}

# Create cleaned labels column
df_threads["sentiment_true_clean"] = df_threads["sentiment_true"].map(label_mapping)

# Check unmapped labels
unmapped = df_threads[df_threads["sentiment_true_clean"].isna()]["sentiment_true"].unique()
print("Unmapped labels:", unmapped)

print(df_threads["sentiment_true_clean"].value_counts())

# Use only manually labeled rows
df_gold = df_threads[df_threads["sentiment_true_clean"].notna()].copy()

# Train/test split on labeled subset
train_df_threads, test_df_threads = train_test_split(
    df_gold,
    test_size=0.2,
    random_state=42,
    stratify=df_gold["sentiment_true_clean"]
)

print("Train distribution:\n", train_df_threads["sentiment_true_clean"].value_counts())
print("Test distribution:\n", test_df_threads["sentiment_true_clean"].value_counts())


# ------------------------------------------------------------
# ----------- SENTIMENT RELIABILITY
# VADER RELIABILITY CHECK WITH CLEAN LABELS (THREADS)
# ------------------------------------------------------------
print("THREADS — VADER SENTIMENT RELIABILITY CHECK")

# Filter only manually labeled rows
df_labeled = df_threads[df_threads['sentiment_true_clean'].notna()].copy()

print(f"\nTotal manually labeled samples: {len(df_labeled)}")
print("\nDistribution of manual labels:")
print(df_labeled['sentiment_true_clean'].value_counts())

print("\nDistribution of VADER predictions:")
print(df_labeled['sentiment'].value_counts())

# Calculate metrics
y_true = df_labeled['sentiment_true_clean']
y_pred = df_labeled['sentiment']

# Overall metrics
accuracy = accuracy_score(y_true, y_pred)
precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

print("\nOVERALL METRICS:")
print(f"Accuracy:          {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision (Macro): {precision_macro:.4f}")
print(f"Recall (Macro):    {recall_macro:.4f}")
print(f"F1-Score (Macro):  {f1_macro:.4f}")

print("\nDETAILED CLASSIFICATION REPORT:")
print(classification_report(y_true, y_pred, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=['negative', 'neutral', 'positive'])

# Row % matrix
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# ------------------------------------------------------------
# PLOT CONFUSION MATRICES
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# COUNTS PLOT
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'],
            ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_title('THREADS — VADER Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted', fontsize=12)
axes[0].set_ylabel('True', fontsize=12)

# PERCENTAGES PLOT
sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'],
            ax=axes[1], cbar_kws={'label': 'Percentage (%)'})
axes[1].set_title('THREADS — VADER Confusion Matrix (Row %)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicted', fontsize=12)
axes[1].set_ylabel('True', fontsize=12)

plt.tight_layout()
plt.show()


# Overall Summary:
# We validated VADER sentiment predictions using 3,047 manually labeled Threads reviews.
# VADER’s overall accuracy was 54.45%, with macro F1 of 0.53, indicating modest performance.
# The confusion matrix shows that VADER tends to overpredict positive sentiment and struggles with neutral or negative statements, largely due to slang and informal writing in user-generated social media data.

# Detailed Analysis:
# To evaluate the reliability of VADER sentiment classification for our Threads dataset, we manually labelled a validation set of 3,047 comments. VADER achieved an accuracy of 54.45% and a macro F1-score of 0.53, indicating moderate but limited performance as a lexicon-based, unsupervised baseline. This is expected because Threads comments contain slang and conversational tone that are difficult for rule-based models to interpret.
# VADER performed well for positive sentiment, achieving a high recall of 92.73% and an F1-score of 0.66, suggesting that it reliably recognises positive language on Threads. However, the model showed substantial challenges with neutral and negative sentiment. Negative comments were identified with reasonable precision (54.02%) but low recall (39.51%), meaning many negative posts were incorrectly classified as neutral or positive. Neutral sentiment was particularly difficult (F1: 0.48), which aligns with known limitations of lexicon-based approaches when dealing with ambiguous, informal, or context-heavy text. Human labelling may also include subjectivity, contributing to inconsistencies in this category.
# Overall, VADER provides a useful initial baseline for sentiment labeling on Threads data, but its limitations—especially around neutral and negative sentiment.

# ------------------------------------------------------------
# Sentiment Classification using TF IDF + Logistic Regression (Threads)
# ------------------------------------------------------------
# Use the full dataset with VADER+KMeans sentiment
df_vader = df_threads.copy()

# Only keep rows where VADER assigned a label (it should be all)
df_vader = df_vader[df_vader["sentiment"].notna()]

print(df_vader["sentiment"].value_counts())

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    min_df=3,
    stop_words="english"
)

# Train-test split
X_train = tfidf.fit_transform(train_df_threads["review_cleaned"])
X_test = tfidf.transform(test_df_threads["review_cleaned"])

y_train = train_df_threads["sentiment"]
y_test = test_df_threads["sentiment"]

# Model Training
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Summary of Results:
# We trained a TF-IDF + Logistic Regression classifier on the Threads dataset using VADER-generated sentiment labels.
# The model achieved an accuracy of 74.43% with a macro F1-score of 0.70.

# Performance varies across sentiment classes:

# - Positive sentiment was detected most reliably (F1 = 0.86) with strong precision (0.93).
# - Neutral class achieved moderate performance (F1 = 0.68) and was generally well-recalled (0.74).
# - Negative sentiment remained the most challenging (F1 = 0.55).

# Overall, these results show that a traditional machine-learning model like TF-IDF + Logistic Regression can still provide strong performance for social media sentiment classification. Despite being trained on weak (auto-generated) labels, the model captures meaningful textual patterns and offers more stable, consistent predictions than lexicon-based approaches such as VADER.

# ------------------------------------------------------------
#
#
################# HAEYEON PLEASE ADD UR DISTILBERT FOR THREADS HERE ###############
#
# ------------------------------------------------------------

# ------------------------------------------------------------
## SENTIMENT ANALYSIS : USING VADER + KNN FOR TWITTER
# ------------------------------------------------------------