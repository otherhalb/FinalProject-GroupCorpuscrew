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
# HYPERPARAMETER TUNING CODE FOR BERT
# ------------------------------------------------------------
# import optuna
# import numpy as np
# from transformers import DistilBertForSequenceClassification, TrainingArguments, Trainer
# from sklearn.metrics import accuracy_score, classification_report

# # Create Validation Set from Training Data
# from sklearn.model_selection import train_test_split

# # Split the training data into train and validation sets
# train_texts = train_df_threads["review_description"].tolist()
# train_labels = y_train_bert

# # Split with stratification to maintain class distribution
# train_texts, val_texts, y_train_split, y_val = train_test_split(
#     train_texts, 
#     train_labels, 
#     test_size=0.2, 
#     random_state=SEED,
#     stratify=train_labels
# )

# # Tokenize the new train split and validation set
# train_enc = tokenizer(
#     train_texts, 
#     truncation=True,
#     padding=True,
#     max_length=128
# )

# val_enc = tokenizer(
#     val_texts,
#     truncation=True,
#     padding=True,
#     max_length=128
# )

# # Create dataset objects for train, validation, and test
# train_dataset = BERTDataset(train_enc, y_train_split)
# val_dataset = BERTDataset(val_enc, y_val)
# test_dataset = BERTDataset(test_enc, y_test_bert)

# # Now you have three datasets:
# # - train_dataset: for training
# # - val_dataset: for hyperparameter tuning (use this in Optuna)
# # - test_dataset: for final evaluation only

# # ------------------------------------------------------------
# # 2. Objective function for Optuna tuning(This will be commented due to avoid running code for 2 hours)
# # ------------------------------------------------------------
# def objective(trial):

#     # ---- Search space ----
#     learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 5e-5)
#     num_epochs = trial.suggest_int('num_train_epochs', 2, 5)
#     batch_size = trial.suggest_categorical('per_device_train_batch_size', [8, 16, 32])
#     weight_decay = trial.suggest_uniform('weight_decay', 0.0, 0.3)
#     warmup_steps = trial.suggest_int('warmup_steps', 0, 500)

#     # ---- New model for every trial ----
#     model = DistilBertForSequenceClassification.from_pretrained(
#         "distilbert-base-uncased",
#         num_labels=3
#     )

#     # ---- Training arguments for this trial ----
#     training_args = TrainingArguments(
#         output_dir=f"./bert_trial_{trial.number}",
#         evaluation_strategy="epoch",
#         save_strategy="no",
#         num_train_epochs=num_epochs,
#         learning_rate=learning_rate,
#         per_device_train_batch_size=batch_size,
#         per_device_eval_batch_size=batch_size,
#         weight_decay=weight_decay,
#         warmup_steps=warmup_steps,
#         seed=SEED,
#         logging_steps=50,
#         load_best_model_at_end=False,
#     )

#     # ---- Trainer ----
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,  # Use the new train_dataset
#         eval_dataset=val_dataset      # Use val_dataset for tuning, NOT test_dataset
#     )

#     # ---- Train ----
#     trainer.train()

#     # ---- Evaluate ----
#     eval_result = trainer.evaluate()

#     # Optuna tries to minimize this value
#     return eval_result["eval_loss"]


# #
# # 3. Run the Optuna study
# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=20)

# print("Best hyperparameters:", study.best_params)
# ------------------------------------------------------------
## SENTIMENT ANALYSIS : USING VADER + KNN FOR TWITTER
# ------------------------------------------------------------

# By applying KMeans clustering to the compound sentiment scores, we could segment reviews into three clusters. 
# The centroids of these clusters inform the thresholds for classifying sentiment, allowing for more objective labeling
# than manual cutoff values.

# ----------- LOAD DATA --------------
df_twitter = twitter_clean.copy()     
# ------------------------------------


# ---------- VADER SCORES -----------
analyzer = SentimentIntensityAnalyzer()
df_twitter["compound"] = df_twitter["review_cleaned"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
compound_array = df_twitter["compound"].values.reshape(-1,1)
# ------------------------------------

# ----------- KMEANS 3 CLUSTERS -----
kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
clusters = kmeans.fit_predict(compound_array)
df_twitter["cluster"] = clusters

# map centroids -> pos/neu/neg
centroids = kmeans.cluster_centers_.flatten()
order = np.argsort(centroids)
label_map = {order[0]: "negative", order[1]: "neutral", order[2]: "positive"}

df_twitter["sentiment"] = df_twitter["cluster"].map(label_map)
# ------------------------------------

print(df_twitter[["review_text","review_cleaned","compound","sentiment"]].head())


df_twitter["sentiment"].value_counts().plot(kind="bar")

plt.title("Sentiment Distribution (KMeans + VADER)")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.show()

df_twitter.boxplot(column="compound", by="sentiment", grid=False)

plt.title("Compound Score by Sentiment Cluster")
plt.suptitle("")
plt.xlabel("Sentiment")
plt.ylabel("Compound Score")
plt.tight_layout()
plt.show()

# Set style
sns.set(style="whitegrid")

# Plot the clusters along the compound score
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=range(len(df_twitter)), 
    y="compound", 
    hue="sentiment", 
    palette={"negative":"red", "neutral":"gray", "positive":"green"},
    data=df_twitter,
    s=50
)
plt.title("KMeans Clusters of Reviews (VADER Compound Scores)")
plt.xlabel("Review Index")
plt.ylabel("Compound Score")
plt.legend(title="Sentiment")
plt.show()

# --------------------------------------------------------------------
# ----------- WORDCLOUDS + TOP WORDS PER SENTIMENT CLUSTER FOR THREADS -----------
# --------------------------------------------------------------------


# Define a function to get top words
def get_top_words(text_series, n=10):
    words = " ".join(text_series).split()
    counter = Counter(words)
    return counter.most_common(n)

# Plot word clouds for each sentiment cluster
sentiments = df_twitter["sentiment"].unique()
plt.figure(figsize=(18,6))

for i, sentiment in enumerate(sentiments, 1):
    text = df_twitter[df_twitter["sentiment"] == sentiment]["review_cleaned"]
    
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

# Summary:
# The negative word cloud shows frustration with Elon/Musk, worsening updates, and strong complaints about 
# the platform becoming “bad,” “worse,” and “ruined.”
# The positive word cloud highlights appreciation for certain features and updates, with users expressing 
# satisfaction through words like good, great, best, and love.
# The neutral word cloud reflects general discussion around Elon/Musk, tweets, changes, and platform updates
# without strong emotional tone.


# --------------------------------------------------------------------
# ----------- TRAIN-TEST SPLIT FOR TWITTER -----------
# --------------------------------------------------------------------


train_df_twitter, test_df_twitter = train_test_split(
    df_twitter, 
    test_size=0.2, 
    random_state=42,
    stratify=df_twitter['sentiment'] 
)
twitter_pos = df_twitter[df_twitter["sentiment"] == "positive"]
twitter_neu = df_twitter[df_twitter["sentiment"] == "neutral"]
twitter_neg = df_twitter[df_twitter["sentiment"] == "negative"]


twitter_pos_train = train_df_twitter[train_df_twitter["sentiment"] == "positive"]
twitter_neu_train = train_df_twitter[train_df_twitter["sentiment"] == "neutral"]
twitter_neg_train = train_df_twitter[train_df_twitter["sentiment"] == "negative"]

twitter_pos_test = test_df_twitter[test_df_twitter["sentiment"] == "positive"]
twitter_neu_test = test_df_twitter[test_df_twitter["sentiment"] == "neutral"]
twitter_neg_test = test_df_twitter[test_df_twitter["sentiment"] == "negative"]


# --------------------------------------------------------------------
# ----------- MAP AND STANDARDIZE MANUAL LABELS -----------
# --------------------------------------------------------------------

# Create mapping dictionary for typos and variations
label_mapping = {
    # Negative variations
    'Negative': 'negative',
    'negative': 'negative',
    
    # Positive variations
    'Positive': 'positive',
    
    # Neutral variations
    'Neutral': 'neutral',
}

# Apply mapping
df_twitter['sentiment_true_clean'] = df_twitter['sentiment_true'].map(label_mapping)

# Check for any unmapped values
unmapped = df_twitter[df_twitter['sentiment_true'].notna() & df_twitter['sentiment_true_clean'].isna()]['sentiment_true'].unique()
if len(unmapped) > 0:
    print(f"\nWARNING: Found unmapped values: {unmapped}")
    print("These will be treated as NaN. Please add them to the mapping if needed.\n")

print("\nAfter cleaning - Value counts for 'sentiment_true_clean':")
print(df_twitter['sentiment_true_clean'].value_counts())

print("\nLabels successfully standardized!")
print(f"  - Total labeled: {df_twitter['sentiment_true_clean'].notna().sum()}")
print(f"  - Negative: {(df_twitter['sentiment_true_clean'] == 'negative').sum()}")
print(f"  - Positive: {(df_twitter['sentiment_true_clean'] == 'positive').sum()}")
print(f"  - Neutral: {(df_twitter['sentiment_true_clean'] == 'neutral').sum()}")


# ----------------------------------------------------------------------------
# ----------- VADER RELIABILITY CHECK WITH CLEAN LABELS FOR TWITTER -----------
# ----------------------------------------------------------------------------

print("VADER SENTIMENT RELIABILITY CHECK")

# Filter non-null sentiment_true values
df_labeled = df_twitter[df_twitter['sentiment_true_clean'].notna()].copy()
print(f"\nTotal manually labeled samples: {len(df_labeled)}")
print(f"\nDistribution of manual labels:")
print(df_labeled['sentiment_true_clean'].value_counts())
print(f"\nDistribution of VADER predictions:")
print(df_labeled['sentiment'].value_counts())

# Calculate metrics
y_true = df_labeled['sentiment_true_clean']
y_pred = df_labeled['sentiment']

# Overall metrics
accuracy = accuracy_score(y_true, y_pred)
precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

print("OVERALL METRICS:")
print(f"Accuracy:          {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision (Macro): {precision_macro:.4f}")
print(f"Recall (Macro):    {recall_macro:.4f}")
print(f"F1-Score (Macro):  {f1_macro:.4f}")

print("DETAILED CLASSIFICATION REPORT:")
print(classification_report(y_true, y_pred, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=['negative', 'neutral', 'positive'])

# Calculate percentages for each row
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'],
            ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_title('VADER Sentiment: Confusion Matrix (Counts)\n(True Labels vs VADER Predicted)', 
                  fontsize=13, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=11)
axes[0].set_xlabel('VADER Predicted', fontsize=11)

# Percentages
sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'],
            ax=axes[1], cbar_kws={'label': 'Percentage (%)'})
axes[1].set_title('VADER Sentiment: Confusion Matrix (Row %)\n(True Labels vs VADER Predicted)', 
                  fontsize=13, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=11)
axes[1].set_xlabel('VADER Predicted', fontsize=11)

plt.tight_layout()
plt.show()

#Summary:  To evaluate the reliability of VADER sentiment classification for our Twitter dataset, 
# we manually labelled a subset of around 3k reviews as a validation set. VADER achieved an overall accuracy of 60.85% 
# with a macro F1-score of 0.54, showing reasonable performance as an unsupervised, lexicon-based baseline for social 
# media sentiment analysis.
#The model worked very well for for negative sentiment with precision (97.25%), indicating high confidence when 
# classifying negative reviews, though with moderate recall (48.95%). Positive sentiment detection was really accurate as well ,
#  with 86.70% recall and an F1-score of 0.77, suggesting VADER effectively captures positive expressions on Twitter data. 
# As expected with rule-based approaches, neutral sentiment proved challenging (F1: 0.19), a common issue when analysing 
# context-dependent langauge in social media. The manual labelling could also be dependent on an individual hence,
#  making it ambiguous for humans as well

#These results validate VADER as a suitable tool for initial sentiment labeling and could be used for sentiment classification. 

# --------------------------------------------------------------------
# --------- TOPIC MODELLING PREPARATION - NMF -----------
# --------------------------------------------------------------------


def run_nmf(docs_df):
    docs_bigrams = prepare_tm_texts(docs_df)
    docs_text = [" ".join(doc) for doc in docs_bigrams]

    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.95,
        min_df=10,
        ngram_range=(1, 1),
    )

    tfidf = tfidf_vectorizer.fit_transform(docs_text)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    dictionary = Dictionary(docs_bigrams)
    dictionary.filter_extremes(no_below=10, no_above=0.5)

    topic_range = range(3, 10)
    coherence_scores = {}

    for k in topic_range:
        nmf_model = NMF(
            n_components=k,
            random_state=42,
            init="nndsvda",
            max_iter=2000
        )
        
        W = nmf_model.fit_transform(tfidf)
        H = nmf_model.components_

        top_words = []
        for topic in H:
            idxs = topic.argsort()[-20:]
            top_words.append([feature_names[i] for i in idxs])

        coherence_model = CoherenceModel(
            topics=top_words,
            texts=docs_bigrams,
            dictionary=dictionary,
            coherence='c_v'
        )

        coherence = coherence_model.get_coherence()
        coherence_scores[k] = coherence
        print(f"k={k} Coherence={coherence:.4f}")

    best_k = max(coherence_scores, key=coherence_scores.get)
    print("\nBest number of NMF topics =", best_k)

    final_nmf = NMF(
        n_components=best_k,
        random_state=42,
        init="nndsvda",
        max_iter=400
    )

    W_final = final_nmf.fit_transform(tfidf)
    H_final = final_nmf.components_

    print("\n--- FINAL NMF TOPIC WORDS ---\n")
    for idx, topic in enumerate(H_final):
        indices = topic.argsort()[-15:]
        words = [feature_names[i] for i in indices]
        print(f"TOPIC {idx+1}: {', '.join(words)}")

    return final_nmf, feature_names



def run_nmf_block(df_input):
    print("\n===== RUNNING NMF BLOCK =====")
    return run_nmf(df_input)


#Visulaizing the Topics in pos, neu, neg (NMF)

# --------------------------------------------------------------------
# VISUALIZATION - LDA
# --------------------------------------------------------------------


def plot_topic(words, weights, title):
    plt.figure(figsize=(10, 5))
    plt.barh(words[::-1], weights[::-1], color='royalblue')
    plt.title(title)
    plt.xlabel("Topic Weight")
    plt.tight_layout()
    plt.show()

def visualize_nmf_topics(nmf_model, feature_names, sentiment_label):
    print(f"\n--- VISUALIZING NMF TOPICS: {sentiment_label.upper()} ---")
    
    H = nmf_model.components_

    for idx, topic in enumerate(H):
        top_idx = topic.argsort()[-15:][::-1]
        words = [feature_names[i] for i in top_idx]
        weights = [topic[i] for i in top_idx]

        title = f"{sentiment_label.upper()} - NMF Topic {idx+1}"
        plot_topic(words, weights, title)

def visualize_nmf_block(nmf_model, feature_names, sentiment_label):
    visualize_nmf_topics(nmf_model, feature_names, sentiment_label)

# --------------------------------------------------------------------
# NMF THREADS ANALYSIS 
# --------------------------------------------------------------------

print("\n===== NMF THREADS POSITIVE =====")
nmf_pos, pos_feats = run_nmf_block(threads_pos)

print("\n===== NMF THREADS NEUTRAL =====")
nmf_neu, neu_feats = run_nmf_block(threads_neu)

print("\n===== NMF THREADS NEGATIVE =====")
nmf_neg, neg_feats = run_nmf_block(threads_neg)

visualize_nmf_block(nmf_pos, pos_feats, "Threads Positive")
visualize_nmf_block(nmf_neu, neu_feats, "Threads Neutral")
visualize_nmf_block(nmf_neg, neg_feats, "Threads Negative")

#Observations:
# Positive Sentiment (Best k = 6) :NMF produced six meaningful topics reflecting various aspects 
# of positive user experience.
# General Impressions - new, use, feature, need
# Interface & Platform Notes - see, ui, platform
# Competitor Comparisons - facebook, instagram, alternative
# Meta Ecosystem Context - social, platform, world
# Functionality Mentions - work, job, start, feature
# Light Positive Reactions - cool, wow, amazing, love

# Neutral Sentiment (Best k = 6): Neutral topics were clearer and more structured than LDA,
#  with well-defined clusters.
# Copying/Clone Commentary - cheap, clone, copied, twitter
# Account & Login Activities - login, delete, create, sign
# Competitor Mentions - mark, zuck, tweeter
# Feed & Usage Observations - feed, see, post, work
# Minor Technical Issues - glitching, bug, not_working
# Routine App Interactions - write, review, comment, time

# Negative Sentiment (Best k = 9): Because negative reviews mention diverse frustrations, 
# NMF discovered nine separate issue clusters, reflecting greater complexity than LDA.
# App Not Working - not_working, glitch, install, ui
# UI & Privacy Concerns - screen, privacy, content
# Missing Functions - post, see, need, delete
# Strong Dissatisfaction - rubbish, pathetic, useless
# Copying/Clone Complaints - copy, copying, cheap, copy_twitter
# Design Issues - boring, nothing_new, poor
# Feed/Discovery Problems - feed, trending, hashtags
# Quality & Competitor Comparison - clone, fake, twitter
# Technical Failures - crash, upload, picture, try


# --------------------------------------------------------------------
# NMF TWITTER ANALYSIS
# --------------------------------------------------------------------


print("\n===== NMF TWITTER POSITIVE =====")
nmf_pos_tw, pos_feats_tw = run_nmf_block(twitter_pos)

print("\n===== NMF TWITTER NEUTRAL =====")
nmf_neu_tw, neu_feats_tw = run_nmf_block(twitter_neu)
print("\n===== NMF TWITTER NEGATIVE =====")
nmf_neg_tw, neg_feats_tw = run_nmf_block(twitter_neg)

visualize_nmf_block(nmf_pos_tw, pos_feats_tw, "Twitter Positive")
visualize_nmf_block(nmf_neu_tw, neu_feats_tw, "Twitter Neutral")
visualize_nmf_block(nmf_neg_tw, neg_feats_tw, "Twitter Negative")

#Observations:
# Positive Sentiment (k ≈ 3) :
# General Enjoyment — love, excellent, best_social
# Feature Appreciation / Improvements — better, awesome, free_speech
# Satisfaction with Updates — good, great, name, change, logo


# Neutral Sentiment (k ≈ 6) :
# Leadership & Feature Mentions — elon_musk, feature
# Platform Behavior — ruined, platform, anymore
# Tweet/Video/Limit Notes — tweet, limit, video
# Rebranding Discussions — bird, back, rebranding
# Update Notes — update, phone
# Logo/Name Change — change, name, logo

# Negative Sentiment (k ≈ 3) :
# Update-Related Frustrations — no, worse, limit, post
# Rebranding Disapproval — suck, name, logo, bird
# Leadership Criticism — bad, elon, ruined, musk

# NMF seems to have decent coherence for positive neutral and negative around 0.40 but 
# it is significantly low than LDA and the insights found in LDA seem to be much better than NMF. 
# Hence, LDA is the best choice for now.

