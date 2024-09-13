import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Set the layout to wide
st.set_page_config(layout="wide")

# Centering the Title and Description with HTML and CSS
st.markdown("""
    <style>
    .centered {
        text-align: center;
    }
    .card {
        padding: 10px; 
        margin: 10px;
        width: 100%; 
        margin-top:40px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);;
        transition: box-shadow 0.3s ease-in-out; /* For smooth hover effect */
    }     
    .card:hover {
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);  /* Elevation effect on hover */
    }
    .small-text {
        font-size: 1em; 
        margin: 0;
    }
            
    .small-text1 {
        font-size: 0.9em; 
        margin: 0;
    }
            
    .sentiment-card {
        width: 60%;
        height: 178px;
        border-radius: 10px; 
        background-color: #ede9f7;
        padding: 10px;
        border-radius: 10px;
        margin-left: 40%;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* Base shadow */
        transition: box-shadow 0.3s ease-in-out; /* Smooth hover */
    }
    .sentiment-card:hover {
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2); /* Elevation on hover */
    }
    .emotion-card {
        width: 60%;
        height: 178px;
        border-radius: 10px; 
        background-color: #ede9f7;
        padding: 15px;
        border-radius: 10px;
        margin-left: 10%;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* Base shadow */
        transition: box-shadow 0.3s ease-in-out; /* Smooth hover */
    }
    .emotion-card:hover {
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2); /* Elevation on hover */
    }
    h5, p {
        margin: 0;
        padding: 0;
    }
            
    h5 {
        padding-top:5px; 
        font-size: 20px;
        color:#482878;
    }
            
    .space-div {
        margin-top: 5%;
    }
            
    h1 {
        padding: 0px;
        padding-left: 1px;
    }
            
    .score {
        padding-top:14px;
        align-items: center;
        font-size: 19px;
        padding-bottom:0px;
    }
            
    .sentiments {
        padding-top:10px;
    }

    </style>
    <div class="centered">
        <h1 style="color:#440154";>Product Review Analysis</h1>
        <p class="sentiments" style="color:#414487";>Analyze customer feedback and gain insights from product reviews.</p>
        <p style="color:#414487";>This dashboard helps visualize the sentiment and emotions expressed in customer reviews.</p>
    </div>
""", unsafe_allow_html=True)


input_file = r"C:\Users\tahiy\VS Code Scripts\Product-Review-Analysis\Data\Output\2Final_sentiment_emotion_analysis.csv"
df = pd.read_csv(input_file)

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])
# Get the date range (min and max)
min_date = df['Date'].min().strftime('%b %d, %Y')
max_date = df['Date'].max().strftime('%b %d, %Y')


# Sentiment and Emotion Counts
sentiment_counts = df['predicted_sentiment'].value_counts()
emotion_counts = df['emotion'].value_counts()

# Calculate sentiment percentages
total_count = len(df)
sentiment_percentages = (sentiment_counts / total_count) * 100
emotion_percentages = (emotion_counts / total_count) * 100

# Define overall sentiment score (assume for demo it's the percentage of positive sentiment)
positive_count = sentiment_counts.get('POSITIVE', 0)
negative_count = sentiment_counts.get('NEGATIVE', 0)
overall_score = round((positive_count - negative_count) / (positive_count + negative_count), 2)
# overall_score = (positive_count - negative_count) / total_count
# overall_score = positive_count / total_count


# Create two columns for Sentiment and Emotion Cards
col1, col2 = st.columns(2)

# Sentiment Score Card
with col1:
    st.markdown(f"""
    <div class="card sentiment-card">
        <div class="bp3-card pt-card pt-interactive pt-elevation-1" style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h5>Customer Satisfaction Score</h5>
                <p style="font-size: 15px; color:#414487;">{min_date} - {max_date}</p>
            </div>
            <h1 style="font-size:40px; color:#482878;">{overall_score}</h1>
        </div>
        <h6 class="score" style="display: flex; align-items: center; justify-content: center; color:#482878;">Sentiment Scores</h6>
        <div class="sentiments" style="display:flex; justify-content: space-evenly;">
            <div>
                <p class="small-text" style="color:#4CAF50; font-weight:bold;">Positive</p>
                <p class="small-text" style="color:#4CAF50;">{sentiment_counts.get('POSITIVE', 0)} | {sentiment_percentages.get('POSITIVE', 0):.0f}%</p>
            </div>
            <div>
                <p class="small-text" style="color:#FFB347; font-weight:bold;">Negative</p>
                <p class="small-text" style="color:#FFB347;">{sentiment_counts.get('NEGATIVE', 0)} | {sentiment_percentages.get('NEGATIVE', 0):.0f}%</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Emotion Score Card
with col2:
    st.markdown(f"""
    <div class="card emotion-card">
        <h6 style="font-size:20px; color:#440154">Emotion Score</h6>
        <div style="display:flex; justify-content:space-between;">
            <div>
                <p class="small-text1" style="color:#4CAF50; font-weight:bold;">Joy😄</p>
                <p class="small-text1" style="color:#4CAF50;">{emotion_counts.get('joy', 0)} | {emotion_percentages.get('joy', 0):.0f}%</p>
            </div>
            <div>
                <p class="small-text1" style="color:#FBC02D; font-weight:bold;">Surprise😲</p>
                <p class="small-text1" style="color:#FBC02D;">{emotion_counts.get('surprise', 0)} | {emotion_percentages.get('surprise', 0):.0f}%</p>
            </div>
            <div>
                <p class="small-text1" style="color:#9E9E9E; font-weight:bold;">Neutral😐</p>
                <p class="small-text1" style="color:#9E9E9E;">{emotion_counts.get('neutral', 0)} | {emotion_percentages.get('neutral', 0):.0f}%</p>
            </div>            
        </div>
        <div style="display:flex; justify-content:space-between;" class="space-div">
            <div>
                <p class="small-text1" style="color:#FFB347; font-weight:bold;">Sadness😢</p>
                <p class="small-text1" style="color:#FFB347;">{emotion_counts.get('sadness', 0)} | {emotion_percentages.get('sadness', 0):.0f}%</p>
            </div>
            <div>
                <p class="small-text1" style="color:#64B5F6; font-weight:bold;">Fear😨</p>
                <p class="small-text1" style="color:#64B5F6;">{emotion_counts.get('fear', 0)} | {emotion_percentages.get('fear', 0):.0f}%</p>
            </div>
            <div>
                <p class="small-text1" style="color:#8D6E63; font-weight:bold;">Disgust🤢</p>
                <p class="small-text1" style="color:#8D6E63;">{emotion_counts.get('disgust', 0)} | {emotion_percentages.get('disgust', 0):.0f}%</p>
            </div>
            <div>
                <p class="small-text1" style="color:#D32F2F; font-weight:bold;">Anger😡</p>
                <p class="small-text1" style="color:#D32F2F;">{emotion_counts.get('anger', 0)} | {emotion_percentages.get('anger', 0):.0f}%</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Next Row - Two Bar Graphs
st.markdown("---")  # Separator

# Load the CSV file
df1 = pd.read_csv(r'C:\Users\tahiy\VS Code Scripts\Product-Review-Analysis\Data\Output\5Final_pros_listed.csv')
df1_1 = pd.read_csv(r'C:\Users\tahiy\VS Code Scripts\Product-Review-Analysis\Data\Output\5Final_cons_listed.csv')

# Extract pros and cons as lists
# pros_list = df1['pros'].dropna().tolist()
pros = df1['pros'].loc[0]
pros_list = pros.strip().split('- ')
# cons_list = df1_1['cons'].dropna().tolist()
cons = df1_1['cons'].loc[0]
cons_list = cons.strip().split('- ')

# Use Streamlit's built-in CSS for full-width columns
st.markdown("""
    <style>
    .full-width-box {
        padding: 0px;
        width: 100%;  
        box-sizing: border-box;
        margin-bottom: 0px;
    }
            
    .centered-header {
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .h3{
        padding-bottom: 0px;
        height:65px;
        background-color: #ede9f7;;
        border-radius: 15px;
        margin-bottom: 0px;
        text-align: center;
        padding-top:14px;
    }
    </style>
    """, unsafe_allow_html=True)

# Centered header above both columns
st.markdown('<div class="centered-header" style="color:#440154";>Areas of Strength and Refinement from Product Reviews</div>', unsafe_allow_html=True)


# Create two full-width columns
col3, col4 = st.columns([1, 1])

# Display Pros in the first column (full width)
with col3:
    st.markdown("""
    <div class="full-width-box">
            <h3 class="h3" style="color:#4CAF50;">Areas of Strengths</h3>
        <ul>
    """, unsafe_allow_html=True)
    
    for pro in pros_list:
        if pro == "":
            continue
        st.markdown(f"<li>{pro}</li>", unsafe_allow_html=True)
    
    st.markdown("</ul></div>", unsafe_allow_html=True)

# Display Cons in the second column (full width)
with col4:
    st.markdown("""
    <div class="full-width-box">
        <h3 class="h3" style="color:#FFB347;">Areas needing Refinement</h3>
        <ul>
    """, unsafe_allow_html=True)
    
    for con in cons_list:
        if con == "":
            continue
        st.markdown(f"<li>{con}</li>", unsafe_allow_html=True)
    
    st.markdown("</ul></div>", unsafe_allow_html=True)


# Next Row - Two Bar Graphs
st.markdown("---")  # Separator


# Custom CSS for cool design and responsiveness
st.markdown("""
    <style>
    .stApp {
        background-color: #F7F7F7;
    }
    .full-width {
        width: 80%;
    }
    .small-title {
        # color: #4CAF50;
        font-size: 30px;
        text-align:center;
    }
    </style>
    """, unsafe_allow_html=True)

# Bar Graphs for Sentiment and Emotion Distribution
col5, col6 = st.columns(2)

# Sentiment Distribution Bar Graph
with col5:
    st.markdown("<h4 class='small-title' style='color:#440154'>Sentiment Distribution</h4>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(4,2))  # Adjusted figure size
    bars = ax.bar(sentiment_counts.index, sentiment_counts.values, color=['#cde11d','#4CAF50'])
    ax.set_xlabel('Sentiment', fontsize=8)
    ax.set_ylabel('Count of Reviews', fontsize=8)
    ax.set_title('Distribution of Sentiments over Customer Reviews', fontsize=5)

    # Add number of reviews on top of each bar
    # for bar in bars:
    #     height = bar.get_height()
    #     ax.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom')
    plt.xticks(fontsize=7)
    # Removing borders from the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    st.pyplot(fig)

# Emotion Distribution Bar Graph
with col6:
    st.markdown("<h4 class='small-title' style='color:#440154'>Emotion Distribution</h4>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(4, 2))  # Adjusted figure size
    ax.bar(emotion_counts.index, emotion_counts.values, 
           color=['#440154', '#443983', '#31688e', '#21918c', '#35b779', '#90d743', '#fde725'])
    ax.set_xlabel('Emotion', fontsize=8)
    ax.set_ylabel('Count of Reviews', fontsize=8)
    ax.set_title('Distribution of Emotions over Customer Reviews', fontsize=5)
    # Rotate x-axis ticks 90 degrees
    plt.xticks(fontsize=7)
    # Removing borders from the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    st.pyplot(fig)

# Define the stop words
stop_words = set(stopwords.words('english'))

# Add custom words you want to remove
custom_words = ['iphone', 'apple', 'amazon', 'phone', '13', 'get']  # Add your own words here
stop_words.update(custom_words)

# Function to remove stop words from a text
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Sample DataFrame
# df['lemma_review_body'] should already exist in your data
df['lemma_review_body_no_stopwords'] = df['lemma_review_body'].apply(remove_stopwords)

# Next Row - Two Bar Graphs
st.markdown("---")  # Separator

# Bar Graphs for Sentiment and Emotion Distribution
col7, col8 = st.columns(2)

with col7:
    st.markdown("<h4 class='small-title' style='color:#440154'>Word Cloud</h4>", unsafe_allow_html=True)
    # Creating text data for WordCloud
    # text_data = ' '.join(df2['word'] * df2['count'])
    text_data = ' '.join(df['lemma_review_body_no_stopwords']).lower()
    
    # Generate Word Cloud
    wordcloud = WordCloud(width=800, height=480, background_color='white').generate(text_data)

    # Display Word Cloud using Matplotlib
    fig, ax = plt.subplots() 
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

with col8:
    st.markdown("<h4 class='small-title' style='color:#440154'>Frequency of Top 10 words</h4>", unsafe_allow_html=True)
    # Tokenize the text into words
    words = text_data.split()
 
    # Count the frequency of each word
    word_counts = Counter(words)

    # Get the top 10 most common words
    top_10_words = word_counts.most_common(10)

    # Convert to a DataFrame for easier plotting
    top_10_df = pd.DataFrame(top_10_words, columns=['word', 'count'])

    # Plot the bar chart using seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(x='count', y='word', data=top_10_df, palette='viridis')
    plt.title('Top 10 Words in Reviews', fontsize=16)
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Words', fontsize=12)

    # Display the plot
    plt.tight_layout()
    st.pyplot(plt)

# Adding a separator line in the Streamlit app
st.markdown("---")  # Separator

# Function to generate n-gram frequency DataFrame
def get_ngram_df(n, df, column='lemma_review_body_no_stopwords'):
    vectorizer = CountVectorizer(ngram_range=(n, n), stop_words='english')
    X = vectorizer.fit_transform(df[column])
    ngrams = vectorizer.get_feature_names_out()
    frequencies = X.sum(axis=0).A1
    ngram_df = pd.DataFrame({'n-gram': ngrams, 'frequency': frequencies})
    return ngram_df.sort_values(by='frequency', ascending=False).head(10)

# Generate unigram, bigram, and trigram data
unigram_df = get_ngram_df(1, df, 'lemma_review_body_no_stopwords')
bigram_df = get_ngram_df(2, df)
trigram_df = get_ngram_df(3, df)

# Function to plot bar charts
def plot_ngram_bar_chart(ngram_df, n, title):
    plt.figure(figsize=(6, 4))
    # plt.barh(ngram_df['n-gram'], ngram_df['frequency'], color='skyblue')
    sns.barplot(y='n-gram', x='frequency', data=ngram_df, palette='viridis')
    plt.xlabel('Frequency')
    plt.ylabel(f'{n}-gram')
    plt.title(title)
    plt.gca().invert_yaxis()
    st.pyplot(plt)

# Display the top 10 unigrams, bigrams, and trigrams with separate plots
col1, col2, col3 = st.columns(3)  # Create three columns for the plots

with col1:
    st.markdown("<h4 class='small-title' style='color:#440154'>Top 10 Unigrams</h4>", unsafe_allow_html=True)
    plot_ngram_bar_chart(unigram_df, 1, 'Top 10 Unigrams')

with col2:
    st.markdown("<h4 class='small-title' style='color:#440154'>Top 10 Bigrams</h4>", unsafe_allow_html=True)
    plot_ngram_bar_chart(bigram_df, 2, 'Top 10 Bigrams')

with col3:
    st.markdown("<h4 class='small-title' style='color:#440154'>Top 10 Trigrams</h4>", unsafe_allow_html=True)
    plot_ngram_bar_chart(trigram_df, 3, 'Top 10 Trigrams')

# tag graph

