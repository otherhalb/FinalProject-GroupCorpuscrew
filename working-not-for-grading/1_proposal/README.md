# FinalProject Proposal-GroupCorpuscrew
## Analyzing Public Sentiment Toward Threads and Twitter Using NLP Techniques

This project aims to analyze and compare user sentiment between Threads and Twitter during their overlapping launch period in July 2023. Threads, launched as a competitor to Twitter, received significant attention and mixed public reactions soon after its release. Understanding how users perceived both platforms during the same period can provide insights into user satisfaction, usability, and platform reception. Using Natural Language Processing (NLP) techniques, the study will classify and compare sentiment trends, identify positive, neutral, and negative opinions, and highlight patterns and suggestions for improvement.

Two publicly available datasets will be used for a fair and time-aligned comparison. The Threads dataset (Threads: An Instagram App Reviews, Kaggle) contains 32,910 reviews from July 2023, and the Twitter dataset (2 Million X (Formerly Twitter) Google Reviews, Kaggle) contains 34,788 reviews from the same month. Both datasets will undergo the same preprocessing and modeling steps to ensure consistent analysis of sentiment trends between platforms.

The project will apply both classical and deep learning–based NLP methods using Python libraries for data processing, modeling, and visualization. The preprocessing pipeline will include lowercasing, punctuation and emoji removal, lemmatization, and stop-word filtering, followed by tokenization. Pandas, numpy, and re will be used for data handling and organization, while scikit-learn, spaCy, and nltk will support feature extraction and text analysis. The baseline model will use TF-IDF with Logistic Regression to detect sentiment polarity, and two deep learning models, Bi-GRU and Bi-GRU-LSTM, will be implemented using PyTorch and torchtext to capture contextual and sequential meaning in user reviews. Visualization tools such as matplotlib, seaborn, and wordcloud will illustrate sentiment patterns and differences between Threads and Twitter. Applying the same pipeline to both datasets will ensure a consistent and fair comparison of sentiment patterns and intensity.

The main NLP tasks include sentiment analysis and sentiment classification. Sentiment analysis will identify frequent words and tones to understand emotional trends, while sentiment classification will categorize reviews as positive, neutral, or negative to compare user attitudes across platforms. Model performance will be evaluated using Accuracy, Precision, Recall, and F1-Score to measure reliability and overall effectiveness. This comparative study aims to uncover user perceptions of both platforms and highlight meaningful differences in sentiment and engagement.

Team (Corpus Crew) Roles:
Haeyeon Jeong: Data Preprocessing and Model 1 (Baseline – TF-IDF + Logistic Regression) 
Snehitha Tadapaneni: Model 2 (Bi-GRU-LSTM) and Evaluation 
Sai Rachana Kandikattu: Model 3 (Bi-GRU) and Evaluation
