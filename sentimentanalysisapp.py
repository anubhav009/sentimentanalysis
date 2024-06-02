import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

st.title("Apple Product Reviews and Twitter Sentiment Analysis")

# File uploaders
uploaded_file_reviews = st.file_uploader("Upload Apple iPhone Reviews CSV", type="csv")
uploaded_file_twitter = st.file_uploader("Upload Apple Twitter Sentiment CSV", type="csv")

if uploaded_file_reviews is not None and uploaded_file_twitter is not None:
    # Load the data
    reviews_df = pd.read_csv(uploaded_file_reviews)
    twitter_df = pd.read_csv(uploaded_file_twitter, encoding='ISO-8859-1')

    # Display the first few rows of each dataframe
    st.subheader("Apple iPhone Reviews Data")
    st.dataframe(reviews_df.head())
    
    st.subheader("Apple Twitter Sentiment Data")
    st.dataframe(twitter_df.head())

    # Visualization: Distribution of Review Ratings
    st.subheader("Distribution of Review Ratings")
    rating_counts = reviews_df['review_rating'].value_counts().sort_index()
    st.bar_chart(rating_counts)
    
    # Visualization: Review Ratings Bar Chart
    st.subheader("Review Ratings Bar Chart")
    fig, ax = plt.subplots()
    ax.bar(rating_counts.index, rating_counts.values)
    ax.set_xlabel('Review Rating')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    # Visualization: Sentiment Distribution Pie Chart
    st.subheader("Sentiment Distribution Pie Chart")
    sentiment_counts = twitter_df['sentiment'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    st.pyplot(fig)

    # Visualization: Box Plot for Sentiment Confidence
    st.subheader("Box Plot for Sentiment Confidence")
    fig, ax = plt.subplots()
    sns.boxplot(x='sentiment', y='sentiment:confidence', data=twitter_df, ax=ax)
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Sentiment Confidence')
    st.pyplot(fig)
    
    # Visualization: Time Series of Review Ratings
    st.subheader("Time Series of Review Ratings")
    reviews_df['reviewed_at'] = pd.to_datetime(reviews_df['reviewed_at'])
    reviews_over_time = reviews_df.set_index('reviewed_at').resample('M')['review_rating'].mean()
    st.line_chart(reviews_over_time)

    # Dropdown for filtering sentiment type for Word Cloud
    sentiment_types = twitter_df['sentiment'].unique()
    selected_sentiment = st.selectbox("Select Sentiment Type for Word Cloud", sentiment_types)
    
    filtered_twitter_df = twitter_df[twitter_df['sentiment'] == selected_sentiment]
    
    # Visualization: Word Cloud for Selected Sentiment
    st.subheader("Word Cloud for Selected Sentiment")
    filtered_twitter_df['text'] = filtered_twitter_df['text'].astype(str).fillna('')
    sentiment_text = " ".join(filtered_twitter_df['text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(sentiment_text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    # Visualization: Sentiment Analysis
    st.subheader("Twitter Sentiment Analysis")
    st.bar_chart(sentiment_counts)
    
    # Visualization: Sentiment Confidence Histogram
    st.subheader("Sentiment Confidence Histogram")
    fig, ax = plt.subplots()
    ax.hist(twitter_df['sentiment:confidence'], bins=20, color='skyblue')
    st.pyplot(fig)
    
    # Visualization: Helpful Count vs. Total Comments
    st.subheader("Helpful Count vs. Total Comments")
    # Handle non-numeric values and convert to integers
    reviews_df['helpful_count'] = reviews_df['helpful_count'].str.replace(' people found this helpful', '').str.replace(',', '')
    reviews_df['helpful_count'] = pd.to_numeric(reviews_df['helpful_count'], errors='coerce').fillna(0).astype(int)
    fig, ax = plt.subplots()
    ax.scatter(reviews_df['helpful_count'], reviews_df['total_comments'])
    ax.set_xlabel('Helpful Count')
    ax.set_ylabel('Total Comments')
    st.pyplot(fig)
    
    # Visualization: Top 10 Words in Reviews
    st.subheader("Top 10 Words in Reviews")
    from collections import Counter
    import re
    words = re.findall(r'\w+', review_text.lower())
    word_counts = Counter(words)
    common_words = word_counts.most_common(10)
    words_df = pd.DataFrame(common_words, columns=['Word', 'Frequency'])
    st.bar_chart(words_df.set_index('Word'))

else:
    st.warning("Please upload both CSV files to proceed.")
