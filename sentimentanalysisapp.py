import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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
    
    # Visualization: Review Ratings Pie Chart
    st.subheader("Review Ratings Pie Chart")
    fig, ax = plt.subplots()
    ax.pie(rating_counts, labels=rating_counts.index, autopct='%1.1f%%')
    st.pyplot(fig)
    
    # Visualization: Word Cloud for Reviews
    st.subheader("Word Cloud for Reviews")
    review_text = " ".join(reviews_df['review_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(review_text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
    
    # Visualization: Sentiment Analysis
    st.subheader("Twitter Sentiment Analysis")
    sentiment_counts = twitter_df['sentiment'].value_counts()
    st.bar_chart(sentiment_counts)
    
    # Visualization: Sentiment Confidence Histogram
    st.subheader("Sentiment Confidence Histogram")
    fig, ax = plt.subplots()
    ax.hist(twitter_df['sentiment:confidence'], bins=20, color='skyblue')
    st.pyplot(fig)
    
    # Dropdown for filtering reviews by country
    countries = reviews_df['review_country'].unique()
    selected_country = st.selectbox("Select Country", countries)
    
    filtered_reviews_df = reviews_df[reviews_df['review_country'] == selected_country]
    st.dataframe(filtered_reviews_df)

    # Visualization: Reviews over time
    st.subheader(f"Reviews Over Time in {selected_country}")
    filtered_reviews_df['reviewed_at'] = pd.to_datetime(filtered_reviews_df['reviewed_at'])
    reviews_over_time = filtered_reviews_df.set_index('reviewed_at').resample('M').size()
    st.line_chart(reviews_over_time)
    
    # Visualization: Helpful Count vs. Total Comments
    st.subheader("Helpful Count vs. Total Comments")
    fig, ax = plt.subplots()
    ax.scatter(reviews_df['helpful_count'].str.replace(' people found this helpful', '').astype(int), 
               reviews_df['total_comments'])
    ax.set_xlabel('Helpful Count')
    ax.set_ylabel('Total Comments')
    st.pyplot(fig)

    # Dropdown for filtering sentiment by query
    queries = twitter_df['query'].unique()
    selected_query = st.selectbox("Select Query", queries)
    
    filtered_twitter_df = twitter_df[twitter_df['query'] == selected_query]
    st.dataframe(filtered_twitter_df)

    # Visualization: Sentiment over time
    st.subheader(f"Sentiment Over Time for {selected_query}")
    filtered_twitter_df['date'] = pd.to_datetime(filtered_twitter_df['date'])
    sentiment_over_time = filtered_twitter_df.set_index('date').resample('M').size()
    st.line_chart(sentiment_over_time)

else:
    st.warning("Please upload both CSV files to proceed.")
