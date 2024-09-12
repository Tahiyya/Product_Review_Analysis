# flag for analysis
import os
import pandas as pd
import openai
import requests
from tqdm import tqdm
import time
import docx

input_file = r'C:\Users\tahiy\VS Code Scripts\Product-Review-Analysis\Data\Output\2Final_sentiment_emotion_analysis.csv'
df = pd.read_csv(input_file)

MODEL_NAME = "phi3:mini"

print("Model client setup initiated.")
client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="nokeyneeded",
)

def generate_pros_cons_list(review, SYSTEM_MESSAGE):

    messages=[
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": review},
    ]

    completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=200,
            n=1,
            # stop=None,
            temperature=0.2
    )
    response_text = completion.choices[0].message.content
    # print("Before",response_text)
    sentiment = [item.strip() for item in response_text.split('-') if item.strip()]
    # print("After",sentiment)
    return sentiment


# Step 1: Filter on 'forAnalysis' column where it is True
df_filtered = df[df['forAnalysis'] == True]

# Step 2: Filter based on predicted_sentiment for 'Positive' and 'Negative'
df_positive = df_filtered[df_filtered['predicted_sentiment'] == 'POSITIVE']
df_negative = df_filtered[df_filtered['predicted_sentiment'] == 'NEGATIVE']

# Analyze the reviews and store the results
pros_list = []

print("Summarizing reviews to generate pros and cons.")
begin_time = time.time()

PROS_SYSTEM_MESSAGE = "You are an AI language model trained to analyze product reviews from customers. Analyze the given review and generate a list of pros mentioned in the review. If no pros are mentioned in the review, return 'None'. Return the response in the form of an unordered list only. Do not include any other explanations."

for index, row in tqdm(df_positive.iterrows(), desc="Processing positive reviews", total=len(df_positive)):
    review = row["review_body"]
    pros = generate_pros_cons_list(review, PROS_SYSTEM_MESSAGE)
    pros_list.append(pros)

# df["summary"] = sentiments
# Step 3: Safely add a new column using .loc to avoid the warning
df_positive.loc[:, "pros_list"] = pros_list

# Analyze the reviews and store the results
cons_list = []

CONS_SYSTEM_MESSAGE = "You are an AI language model trained to analyze product reviews from customers. Analyze the given review and generate a list of cons mentioned in the review. If no cons are mentioned in the review, return 'None'. Return the response in the form of an unordered list only. Do not include any other explanations."

print("Summarizing reviews to generate pros and cons.")
begin_time = time.time()

for index, row in tqdm(df_negative.iterrows(), desc="Processing negative reviews", total=len(df_negative)):
    review = row["review_body"]
    cons = generate_pros_cons_list(review, CONS_SYSTEM_MESSAGE)
    cons_list.append(cons)

# df["summary"] = sentiments
df_negative.loc[:, "cons_list"] = cons_list

print("Reviews Summarized")
finish_time = time.time()
execution_time = finish_time - begin_time
print(f"Total Execution time: {execution_time:.4f} seconds")

path = r"C:\Users\tahiy\VS Code Scripts\Product-Review-Analysis\Data\Output"
# Step 3: Save both DataFrames to CSV
df_positive.to_csv(path + r'\3Final_summarized_pros.csv', index=False)
df_negative.to_csv(path + r'\3Final_summarized_cons.csv', index=False)





