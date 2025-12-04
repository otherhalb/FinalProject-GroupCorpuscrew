# -----------------------------------------------------------
# My Code File: Snehitha Tadapaneni
# -----------------------------------------------------------
# I have worked on the below code using some references from the internet: references mentioned at the end.
# Contents:
# 1. Manual Labelling and Comparision with VADER Sentiments
# 2. Logistic Regression + TF-IDF
# Topic Modelling: LDA
# -----------------------------------------------------------
#
# ------------------------------------------------------
## SENTIMENT ANALYSIS : USING VADER + KNN FOR THREADS
# ------------------------------------------------------
# By applying KMeans clustering to the compound sentiment scores, we could segment reviews into three clusters. The centroids of these clusters inform the thresholds for classifying sentiment, allowing for more objective labeling than manual cutoff values.

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

# ------------------------------------------------------------------------------------------------
# ----------- TOPIC MODELLING WITH LDA  -----------
# ------------------------------------------------------------------------------------------------
# Data Cleaning 2
negations = {"no", "not", "nor", "dont", "can't", "cannot", "never"}
stop_words = stop_words - negations

# App-specific words to remove
custom_stopwords_tm = set([
    "app", "apps", "application", "applications",
    "experience", "account", 
])

# ============================
#  Cleaning Function
# ============================
def clean_text_for_tm(text):  
    tokens = simple_preprocess(text, deacc=True)
    return tokens

# ============================
#  Bigram Builder
# ============================
def create_bigrams(texts):
    bigram = Phrases(texts, min_count=10, threshold=10)
    bigram_mod = Phraser(bigram)
    return [bigram_mod[doc] for doc in texts]

# ============================
# Prepare Dataset for Topic Modeling
# ============================
def prepare_tm_texts(df):
    docs = df["review_cleaned"].apply(clean_text_for_tm).tolist()
    docs_bigrams = create_bigrams(docs)
    return docs_bigrams

def train_lda_model(docs_bigrams, num_topics=5):

    dictionary = Dictionary(docs_bigrams)
    dictionary.filter_extremes(no_below=10, no_above=0.5)

    corpus = [dictionary.doc2bow(doc) for doc in docs_bigrams]

    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        chunksize=2000,
        passes=10,
        alpha='auto'
    )

    coherence_model = CoherenceModel(
        model=lda_model,
        texts=docs_bigrams,
        dictionary=dictionary,
        coherence='c_v'
    )

    coherence_score = coherence_model.get_coherence()

    return lda_model, corpus, dictionary, coherence_score



def run_lda(docs_df):
    docs_bigrams = prepare_tm_texts(docs_df)

    topic_range = range(3, 7)
    coherence_scores = {}

    for k in topic_range:
        lda_model, corpus, dictionary, coherence = train_lda_model(docs_bigrams, num_topics=k)
        coherence_scores[k] = coherence
        print(f"k={k} Coherence={coherence:.4f}")

    best_k = max(coherence_scores, key=coherence_scores.get)
    print("\nBest number of topics =", best_k)

    final_lda, corpus, dictionary, coherence = train_lda_model(docs_bigrams, num_topics=best_k)

    print("Final Coherence Score:", coherence)

    topics = final_lda.show_topics(num_topics=-1, num_words=15, formatted=False)

    for topic_id, words in topics:
        print(f"TOPIC {topic_id+1}:")
        top_terms = [w for w, weight in words]
        print(", ".join(top_terms))
        print()

    return final_lda    # 


print("\n===== POSITIVE TOPICS =====")
def run_lda_block(df_input):
    print("\n===== RUNNING LDA BLOCK =====")
    return run_lda(df_input)

# print("\n===== THREADS NEUTRAL TOPICS =====")
# lda_neu = run_lda(threads_neu)

# print("\n===== THREADS NEGATIVE TOPICS =====")
# lda_neg = run_lda(threads_neg)

# Visualize Topics
import matplotlib.pyplot as plt
import math

# ---------------------------------------------------------
# 4×4 Grid Visualization for One Sentiment
# ---------------------------------------------------------
def visualize_lda_grid(lda_model, sentiment_label, num_words=10):

    topics = lda_model.show_topics(num_topics=-1, num_words=num_words, formatted=False)
    num_topics = len(topics)

    print(f"\n--- VISUALIZING {num_topics} TOPICS FOR {sentiment_label.upper()} ---")

    # Grid size (max 4x4)
    rows = 4
    cols = 4
    total_plots = rows * cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, 18))
    axes = axes.flatten()

    for i, ax in enumerate(axes):

        if i < num_topics:
            topic_id, topic_data = topics[i]
            words = [w for w, wt in topic_data]
            weights = [wt for w, wt in topic_data]

            ax.barh(words[::-1], weights[::-1])
            ax.set_title(f"Topic {topic_id+1}")
            ax.tick_params(labelsize=9)
        else:
            ax.axis("off")   # hide empty grid cells

    plt.suptitle(f"{sentiment_label.upper()} — LDA Topics", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

# ---------------------------------------------------------
# For Threads
# ---------------------------------------------------------
def visualize_lda_block(lda_model, sentiment_label):
    visualize_lda_grid(lda_model, sentiment_label)

print("\n===== THREADS POSITIVE TOPICS =====")
lda_pos_threads = run_lda_block(threads_pos)
print("\n===== THREADS NEUTRAL TOPICS =====")
lda_neu_threads = run_lda_block(threads_neu)
print("\n===== THREADS NEGATIVE TOPICS =====")
lda_neg_threads = run_lda_block(threads_neg)

visualize_lda_block(lda_pos_threads, "Threads Positive")
visualize_lda_block(lda_neu_threads, "Threads Neutral")
visualize_lda_block(lda_neg_threads, "Threads Negative")

# Threads Analysis:
#  - Positive reviews: Users express generally positive impressions and note that the app works well, while also offering polite suggestions for additional features.
#  - Neutral reviews: Comments focus on platform comparisons, technical observations, and practical usage notes without strong emotional tone.
#  - Negative reviews: Users report dissatisfaction related to crashes, missing features, feed issues, and comparisons where Threads feels weaker than alternatives.

# -------------------------------------------------------
# For Twitter
# -------------------------------------------------------  
print("\n===== TWITTER POSITIVE TOPICS =====")
lda_pos_twitter = run_lda_block(twitter_pos)

print("\n===== TWITTER NEUTRAL TOPICS =====")
lda_neu_twitter = run_lda_block(twitter_neu)

print("\n===== TWITTER NEGATIVE TOPICS =====")
lda_neg_twitter = run_lda_block(twitter_neg)

visualize_lda_block(lda_pos_twitter, "Twitter Positive")
visualize_lda_block(lda_neu_twitter, "Twitter Neutral")
visualize_lda_block(lda_neg_twitter, "Twitter Negative")

# Twitter Analysis:
#  - Positive reviews: Users express favorable impressions, highlight improvements, and appreciate specific features or platform behavior.
#  - Neutral reviews: Comments focus on platform changes, updates, and general functionality, often mentioning rebranding or feature adjustments without strong emotion.
#- Negative reviews: Users express dissatisfaction with updates, rebranding decisions, and limits, frequently criticizing how recent changes have affected the platform experience.

# References:
# 1. https://scikit-learn.org/stable/modules/clustering.html
# 2. https://github.com/keitazoumana/lda-tutorial/blob/main/topic-modeling-with-lda.ipynb
# 3. https://www.kaggle.com/code/kashnitsky/logistic-regression-tf-idf-baseline