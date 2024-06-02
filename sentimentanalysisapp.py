import streamlit as st
import pandas as pd

# Function to load data and ensure 'date' column exists
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
    else:
        st.warning("Uploaded CSV does not contain a 'date' column.")
        data['date'] = pd.NaT  # Assign Not-A-Time to date column
    return data

# Title
st.title("Sentiment Analysis of Apple iPhone Reviews")

# Sidebar
st.sidebar.header("User Input Features")

# File upload
uploaded_amazon_file = st.sidebar.file_uploader("Upload Amazon Reviews CSV", type=["csv"])
uploaded_twitter_file = st.sidebar.file_uploader("Upload Twitter Sentiment CSV", type=["csv"])

# Load data
if uploaded_amazon_file is not None:
    amazon_data = load_data(uploaded_amazon_file)

if uploaded_twitter_file is not None:
    twitter_data = load_data(uploaded_twitter_file)

# Dataset selection
if uploaded_amazon_file is not None and uploaded_twitter_file is not None:
    selected_dataset = st.sidebar.selectbox('Select Dataset', ('Amazon', 'Twitter'))

    # Sentiment selection
    sentiment_options = ['Positive', 'Neutral', 'Negative']
    selected_sentiment = st.sidebar.multiselect('Select Sentiment', sentiment_options, sentiment_options)

    # Date range selection
    if selected_dataset == 'Amazon':
        data = amazon_data
    else:
        data = twitter_data

    # Filter out rows without valid dates
    if data['date'].isna().all():
        st.warning(f"The selected dataset ({selected_dataset}) does not contain valid date information.")
    else:
        data = data.dropna(subset=['date'])

        # Date range slider
        date_range = st.sidebar.slider(
            'Select Date Range',
            min_value=data['date'].min().date(),
            max_value=data['date'].max().date(),
            value=(data['date'].min().date(), data['date'].max().date())
        )

        # Filter data based on user input
        filtered_data = data[
            (data['sentiment'].isin(selected_sentiment)) & 
            (data['date'] >= pd.to_datetime(date_range[0])) & 
            (data['date'] <= pd.to_datetime(date_range[1]))
        ]

        # Display filtered data
        st.write(f"Displaying {selected_dataset} reviews with {selected_sentiment} sentiment from {date_range[0]} to {date_range[1]}")
        st.write(filtered_data.head())

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
