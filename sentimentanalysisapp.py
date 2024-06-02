import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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
    
    # Visualization: Sentiment Analysis
    st.subheader("Twitter Sentiment Analysis")
    sentiment_counts = twitter_df['sentiment'].value_counts()
    st.bar_chart(sentiment_counts)
    
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
