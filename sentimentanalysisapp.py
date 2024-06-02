import streamlit as st
import pandas as pd
import plotly.express as px

# Function to load and preprocess Amazon data
def load_amazon_data(uploaded_file):
    data = pd.read_csv(uploaded_file, encoding='utf-8')
    data.rename(columns={'reviewed_at': 'date'}, inplace=True)
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    
    # Map review ratings to sentiment
    rating_to_sentiment = {
        '5.0 out of 5 stars': 'Positive',
        '4.0 out of 5 stars': 'Positive',
        '3.0 out of 5 stars': 'Neutral',
        '2.0 out of 5 stars': 'Negative',
        '1.0 out of 5 stars': 'Negative'
    }
    data['sentiment'] = data['review_rating'].map(rating_to_sentiment)
    
    return data

# Function to load and preprocess Twitter data
def load_twitter_data(uploaded_file):
    data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    
    sentiment_map = {0: 'Negative', 2: 'Neutral', 4: 'Positive'}
    data['sentiment'] = data['sentiment'].map(sentiment_map)
    
    return data

# Title
st.title("Sentiment Analysis of Apple iPhone Reviews")

# Sidebar
st.sidebar.header("User Input Features")

# File upload
uploaded_amazon_file = st.sidebar.file_uploader("Upload Amazon Reviews CSV", type=["csv"])
uploaded_twitter_file = st.sidebar.file_uploader("Upload Twitter Sentiment CSV", type=["csv"])

# Load data
amazon_data = twitter_data = None
if uploaded_amazon_file is not None:
    amazon_data = load_amazon_data(uploaded_amazon_file)

if uploaded_twitter_file is not None:
    twitter_data = load_twitter_data(uploaded_twitter_file)

# Dataset selection
if amazon_data is not None and twitter_data is not None:
    selected_dataset = st.sidebar.selectbox('Select Dataset', ('Amazon', 'Twitter'))

    # Sentiment selection
    sentiment_options = ['Positive', 'Neutral', 'Negative']
    selected_sentiment = st.sidebar.multiselect('Select Sentiment', sentiment_options, sentiment_options)

    # Date range selection
    if selected_dataset == 'Amazon':
        data = amazon_data
    else:
        data = twitter_data

    if not selected_sentiment:
        st.warning("Please select at least one sentiment.")
    else:
        # Ensure the date range slider values are datetime.date objects
        min_date = data['date'].min().date()
        max_date = data['date'].max().date()
        
        date_range = st.sidebar.slider(
            'Select Date Range',
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date)
        )

        # Convert date_range to datetime
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])

        # Filter data based on user input
        filtered_data = data[
            (data['sentiment'].isin(selected_sentiment)) & 
            (data['date'] >= start_date) & 
            (data['date'] <= end_date)
        ]

        # Display filtered data
        st.write(f"Displaying {selected_dataset} reviews with {selected_sentiment} sentiment from {start_date.date()} to {end_date.date()}")
        st.write(filtered_data.head())

        # Plot sentiment distribution over time
        fig = px.histogram(filtered_data, x='date', color='sentiment', title='Sentiment Distribution Over Time')
        st.plotly_chart(fig)

        # Plot sentiment count
        sentiment_count = filtered_data['sentiment'].value_counts().reset_index()
        sentiment_count.columns = ['sentiment', 'count']
        fig = px.bar(sentiment_count, x='sentiment', y='count', color='sentiment', title='Sentiment Count')
        st.plotly_chart(fig)

    # Model selection
    model_options = ['Logistic Regression', 'Naive Bayes', 'XGBClassifier', 'BERT']
    selected_model = st.sidebar.selectbox('Select Model', model_options)

    # Display model information (dummy content for illustration)
    st.subheader("Selected Model Information")
    if selected_model == 'Logistic Regression':
        st.write("Logistic Regression is a linear model for binary classification predictive modeling.")
    elif selected_model == 'Naive Bayes':
        st.write("Naive Bayes is a probabilistic classifier based on Bayes' theorem with strong independence assumptions.")
    elif selected_model == 'XGBClassifier':
        st.write("XGBClassifier is an implementation of gradient boosted decision trees designed for speed and performance.")
    elif selected_model == 'BERT':
        st.write("BERT is a transformer-based model designed to understand the context of a word in search queries.")

    # Placeholder for additional visualizations or model results
    st.subheader("Additional Visualizations and Model Results")
    st.write("Visualizations and results will be displayed here based on the selected filters and model.")

else:
    st.write("Please upload both Amazon and Twitter dataset CSV files to proceed.")

# End of the app
st.write("End of analysis.")
