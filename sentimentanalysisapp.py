import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# File uploaders
uploaded_file_reviews = st.file_uploader("Upload Apple iPhone Reviews CSV", type="csv")
uploaded_file_twitter = st.file_uploader("Upload Apple Twitter Sentiment CSV", type="csv")

if uploaded_file_reviews and uploaded_file_twitter:
    # Load data
    iphone_reviews = pd.read_csv(uploaded_file_reviews)
    twitter_sentiment = pd.read_csv(uploaded_file_twitter, encoding='ISO-8859-1')

    # Data wrangling
    iphone_reviews['review_rating'] = iphone_reviews['review_rating'].str.extract('(\d+\.\d+)').astype(float)
    iphone_reviews['reviewed_at'] = pd.to_datetime(iphone_reviews['reviewed_at'])

    twitter_sentiment = twitter_sentiment[twitter_sentiment['sentiment'] != 'not_relevant']
    twitter_sentiment['sentiment'] = pd.to_numeric(twitter_sentiment['sentiment'], errors='coerce')
    valid_twitter_sentiment = twitter_sentiment[twitter_sentiment['sentiment'].isin([1, 2, 3])]

    sentiment_mapping = {1: 'Positive', 2: 'Neutral', 3: 'Negative'}
    valid_twitter_sentiment['sentiment_label'] = valid_twitter_sentiment['sentiment'].map(sentiment_mapping)
    valid_twitter_sentiment['date'] = pd.to_datetime(valid_twitter_sentiment['date'])

    # Streamlit app
    st.title('Apple iPhone Reviews and Twitter Sentiment Analysis')

    st.header('DataFrames')
    st.subheader('iPhone Reviews')
    st.write(iphone_reviews.head())

    st.subheader('Twitter Sentiment')
    st.write(valid_twitter_sentiment.head())

    st.header('Visualizations')

    # Distribution of Review Ratings
    st.subheader('Distribution of Review Ratings')
    fig, ax = plt.subplots()
    iphone_reviews['review_rating'].hist(bins=20, edgecolor='black', ax=ax)
    ax.set_title('Distribution of Review Ratings')
    ax.set_xlabel('Review Rating')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Review Ratings Bar Chart
    st.subheader('Review Ratings Bar Chart')
    fig, ax = plt.subplots()
    iphone_reviews['review_rating'].value_counts().sort_index().plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title('Review Ratings Bar Chart')
    ax.set_xlabel('Review Rating')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Sentiment Distribution
    st.subheader('Distribution of Sentiments')
    fig, ax = plt.subplots()
    valid_twitter_sentiment['sentiment_label'].value_counts().plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title('Distribution of Sentiments')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Reviews Over Time
    st.subheader('Number of Reviews Over Time')
    fig, ax = plt.subplots()
    iphone_reviews.set_index('reviewed_at').resample('M').size().plot(ax=ax)
    ax.set_title('Number of Reviews Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Reviews')
    st.pyplot(fig)

    # Filter reviews by star ratings
    st.subheader('Filter Reviews by Star Rating')
    star_rating = st.selectbox('Select Star Rating', sorted(iphone_reviews['review_rating'].unique()))
    filtered_reviews = iphone_reviews[iphone_reviews['review_rating'] == star_rating]
    st.write(filtered_reviews)

    # Plot for filtered reviews by star rating
    st.subheader(f'Number of Reviews for {star_rating} Star Rating Over Time')
    fig, ax = plt.subplots()
    filtered_reviews.set_index('reviewed_at').resample('M').size().plot(ax=ax)
    ax.set_title(f'Number of Reviews for {star_rating} Star Rating Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Reviews')
    st.pyplot(fig)

    # Model Accuracy Plot
    st.subheader('Model Accuracy')
    model_types = ['LogisticRegression', 'Multinomial Naive Bayes classifier', 'XGBClassifier', 'BERT']
    accuracy_scores = [0.7552950075642966, 0.7299546142208775, 0.7409228441754917, 0.7744811489219366]
    selected_model = st.selectbox('Select Model', model_types)
    
    fig, ax = plt.subplots()
    ax.plot(model_types, accuracy_scores, marker='o', linestyle='-')
    
    for i, (xi, yi) in enumerate(zip(model_types, accuracy_scores)):
        ax.annotate(f'({xi}, {yi:.4f})', (xi, yi), textcoords="offset points", xytext=(0, 15), ha='center')
    
    ax.set_title('Model Accuracy')
    ax.set_xlabel('Model Name')
    ax.set_ylabel('Accuracy Score')
    ax.grid(True)
    st.pyplot(fig)

    # Generate Word Cloud
    def generate_wordcloud(sentiment_label=None):
        if sentiment_label:
            text = " ".join(review for review in valid_twitter_sentiment[valid_twitter_sentiment['sentiment_label'] == sentiment_label]['text'])
            if text:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title(f'Word Cloud for {sentiment_label} Sentiment')
                st.pyplot(fig)
            else:
                st.write(f"No text data for {sentiment_label} sentiment.")

    st.subheader('Word Cloud for Reviews')
    sentiment_type = st.selectbox('Select Sentiment Type', ['Positive', 'Negative'])
    generate_wordcloud(sentiment_type)
else:
    st.write("Please upload both the Apple iPhone Reviews CSV and the Apple Twitter Sentiment CSV.")
