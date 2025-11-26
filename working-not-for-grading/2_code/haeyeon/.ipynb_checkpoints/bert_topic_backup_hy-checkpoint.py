embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(texts, show_progress_bar=True)

# Disable UMAP for stability (macOS + Python 3.11)
bertopic_model = BERTopic(
    language="english",
    umap_model=None, # Disable UMAP
    min_topic_size=20,
    calculate_probabilities=True,
    verbose=True
)

# Fit model
bertopic_topics, bertopic_probs = bertopic_model.fit_transform(texts, embeddings)

df["bertopic_topic"] = bertopic_topics
df["bertopic_prob"] = bertopic_probs.tolist()

print("\n--- Raw BERTopic Summary ---")
print(bertopic_model.get_topic_info().head())

# Reduce topics to improve readability
print("\nReducing topics to around 25")
bertopic_model = bertopic_model.reduce_topics(texts, nr_topics=25)

# Re-transform to generate the updated topic assignments
new_topics, new_probs = bertopic_model.transform(texts)

df["bertopic_topic"] = new_topics
df["bertopic_prob"] = new_probs

# Retrieve updated topic information
topic_info = bertopic_model.get_topic_info()
print("\n--- Reduced Topic Info ---")
print(topic_info)

# Top words per topic
print("\n--- Top Words per Topic ---")
for topic_id in topic_info["Topic"].unique():
    print(f"\nTopic {topic_id}:")
    print(bertopic_model.get_topic(topic_id))

# Topic Distribution Plot
plt.figure(figsize=(10,6))
sns.barplot(x="Topic", y="Count", data=topic_info)
plt.title("BERTopic — Topic Count Distribution")
plt.tight_layout()
plt.show()

# Representative Documents per Topic
for topic_id in topic_info["Topic"].unique():
    print(f"\n--- Representative Docs for Topic {topic_id} ---")

    # Correct method:
    docs_dict = bertopic_model.get_representative_docs(topic_id)
    docs = docs_dict.get(topic_id, [])  # FIX HERE ✔

    for d in docs[:3]:
        print("-", d[:200])  # First 200 characters