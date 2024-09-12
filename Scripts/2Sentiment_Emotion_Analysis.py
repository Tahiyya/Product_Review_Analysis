import pandas as pd
from transformers import pipeline, AutoTokenizer
import numpy as np
import torch
from datasets import Dataset


inputFilepath = r"C:\Users\tahiy\VS Code Scripts\Product-Review-Analysis\Data\Output\1Final_Transformed.csv"
df = pd.read_csv(inputFilepath)

# Check if GPU is available, and set device accordingly (0 for GPU, -1 for CPU)
device = 0 if torch.cuda.is_available() else -1

print("Setting up Sentiment and Emotion Analysis Models.")

# Initialize tokenizer and sentiment analysis pipeline
model_name = 'bert-base-uncased'  # Use the model name appropriate for your task
tokenizer = AutoTokenizer.from_pretrained(model_name)
# sentiment_pipeline = pipeline('sentiment-analysis', model="siebert/sentiment-roberta-large-english", tokenizer=tokenizer, device=device)
# Load pre-trained models
sentiment_pipeline = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english", device=device)
# sentiment_pipelin = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english', device=device)
emotion_pipeline = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', top_k=1, device=device)

# Convert DataFrame to Dataset
dataset = Dataset.from_pandas(df)

# Define functions for sentiment and emotion detection
def get_sentiment(batch):
    texts = batch['cleaned_review_body']
    # texts = bath['lemma_review_body']
    results = sentiment_pipeline(texts)
    sentiments = [result['label'] for result in results]
    scores = [result['score'] for result in results]
    return {'predicted_sentiment': sentiments, 'predicted_sentiment_score': scores}

# def get_sentiment(batch):
#     # Extract texts from the batch
#     texts = batch['cleaned_review_body']
    
#     # Tokenize and truncate texts
#     max_length = 512
#     inputs = tokenizer(texts, 
#                        truncation=True, 
#                        padding=True, 
#                        max_length=max_length, 
#                        return_tensors='pt')  # Optionally include attention_mask
    
#     # Get sentiment analysis results
#     results = sentiment_pipeline(inputs['input_ids'])
    
#     # Extract sentiments and scores
#     sentiments = [result['label'] for result in results]
#     scores = [result['score'] for result in results]
    
#     return {'predicted_sentiment': sentiments, 'predicted_sentiment_score': scores}

def get_emotions(batch):
    # {'disgust': None, 'fear': None, 'joy': None, 'neutral': None, 'sadness': None, 'surprise': None}
    texts = batch['cleaned_review_body']
    # texts = bath['lemma_review_body']
    results = emotion_pipeline(texts)
    emotions = []
    for result in results:
        filtered_emotions = {emotion['label']: emotion['score'] for emotion in result  if emotion['score'] >= 0.7}
        emotions.append(filtered_emotions)
    return {'predicted_emotions': emotions}


# Apply functions to dataset
print("Analyzing sentiments")
dataset = dataset.map(get_sentiment, batched=True)

print("Analyzing emotions")
dataset = dataset.map(get_emotions, batched=True)

# Convert Dataset back to DataFrame
df = dataset.to_pandas()

# Assuming the column with the dictionary is named 'predicted_emotions'
def filter_emotions(emotion_dict):
    # Filter out keys (emotions) where the value is None
    emotion_list = [emotion for emotion, score in emotion_dict.items() if score is not None]
    if len(emotion_list) == 1:
        return emotion_list[0]
    else:
        return ""

# Apply the function to each row in the 'predicted_emotions' column
df['emotion'] = df['predicted_emotions'].apply(filter_emotions)

print("Analysis completed.")
outputFilePath = r'C:\Users\tahiy\VS Code Scripts\Product-Review-Analysis\Data\Output\2Final_sentiment_emotion_analysis.csv'
df.to_csv(outputFilePath)

