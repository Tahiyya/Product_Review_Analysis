import os
import pandas as pd
import openai
import requests
from tqdm import tqdm
import time
import ast

MODEL_NAME = "phi3:mini"

print("Client setup")
client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="nokeyneeded",
)

# Helper function to count the number of words in a list of strings
def word_count(text):
    return len(text.split())

# Function to split the combined pros list into chunks based on word limit
def split_by_word_limit(pros_list, word_limit=1800):
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for pros in pros_list:
        pros_word_count = word_count(pros)
        
        # If adding this pros exceeds the word limit, save the current chunk
        if current_word_count + pros_word_count > word_limit:
            chunks.append(current_chunk)
            current_chunk = []
            current_word_count = 0
        
        # Add the pros to the current chunk
        current_chunk.append(pros)
        current_word_count += pros_word_count
    
    # Append the last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# The code is designed to read 1,800 or less words data at a time in the list form and determine the pros and cons people have mentioned from that block of text. 
# The code will then move onto the next set of 1,800 words and extract the pros and cons from it, repeating as necessary until all of the reviews have been processed.

# Generate a list of pros and cons from all of the raw user reviews
def generate_proscons_list(text, SYSTEM_MESSAGE):
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": text}]

    completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=300,
            n=1,
            # stop=None,
            temperature=0.2
    )

    response = completion.choices[0].message.content
    # print("Response:", response)
    return response


# Read the reviews data from the CSV input file and then create a dataframe to hold the review data

input_file1 = r"C:\Users\tahiy\VS Code Scripts\Product-Review-Analysis\Data\Output\3Final_summarized_pros.csv"
input_file2 = r"C:\Users\tahiy\VS Code Scripts\Product-Review-Analysis\Data\Output\3Final_summarized_cons.csv"

df = pd.read_csv(input_file1)
df2 = pd.read_csv(input_file2)

df['pros_list'] = df['pros_list'].apply(ast.literal_eval)
df2['cons_list'] = df2['cons_list'].apply(ast.literal_eval)

# Flatten the pros_list into a single list of strings
combined_pros_list = [item for sublist in df['pros_list'] for item in sublist]
combined_cons_list = [item for sublist in df2['cons_list'] for item in sublist]
# print(combined_pros_list)

# Chunk the combined pros list into approximately 1,800-word chunks
chunked_pros_lists = split_by_word_limit(combined_pros_list, word_limit=1800)
chunked_cons_lists = split_by_word_limit(combined_cons_list, word_limit=1800)
# print(chunked_pros_lists)

list_proscons = []
list_proscons2 = []

PROS_SYSTEM_MESSAGE = "Customer reviews for the iPhone 13 are given separated by a semi-colon (;) as delimiter. Generate one consolidated list by removing duplicates and keeping the most relevant pros. Provide one list with unique pros prioritized only the most frequently mentioned pros. Return the response in the form of an unordered list only. Do not include any other explanations."
CONS_SYSTEM_MESSAGE = "Customer reviews for the iPhone 13 are given separated by a semi-colon (;) as delimiter. Generate one consolidated list by removing duplicates and keeping the most relevant cons. Provide one list with unique cons prioritized only the most frequently mentioned cons. Return the response in the form of an unordered list only. Do not include any other explanations."


# Pass each chunk to the SLM Phi3:Mini model
for chunk in chunked_pros_lists:
    print("\n\n--------------------------------------------Chunk")
    combined_chunk = "; ".join(chunk)  # Combine the chunk into a single string
    # print("\n\n--------------------------------------------Combined_chunk", combined_chunk)

    unique_pros_list = generate_proscons_list(combined_chunk, PROS_SYSTEM_MESSAGE) 
    print("\n\n--------------------------------------------Response")
    # print("\n\n--------------------------------------------unique pros_list", unique_pros_list)
    list_proscons.append(unique_pros_list)


# # Save the resulting list of pros  to a new Excel file for further offline processing
df_proscons = pd.DataFrame()
df_proscons["pros_cons"] = list_proscons
output_file_proscons = r"C:\Users\tahiy\VS Code Scripts\Product-Review-Analysis\Data\Output\4Final_pros_list.csv"
df_proscons.to_csv(output_file_proscons, index=False)

# Pass each chunk to the SLM Phi3:Mini model
for chunk in chunked_cons_lists:
    print("\n\n--------------------------------------------Chunk")
    combined_chunk = "; ".join(chunk)  # Combine the chunk into a single string
    # print("\n\n--------------------------------------------Combined_chunk", combined_chunk)

    unique_cons_list = generate_proscons_list(combined_chunk, CONS_SYSTEM_MESSAGE) 
    print("\n\n--------------------------------------------Response")
    # print("\n\n--------------------------------------------unique pros_list", unique_pros_list)
    list_proscons2.append(unique_cons_list)


# Save the resulting list of pros and cons to a new Excel file for further offline processing
df_proscons2 = pd.DataFrame()
df_proscons2["pros_cons"] = list_proscons2
output_file_proscons2 = r"C:\Users\tahiy\VS Code Scripts\Product-Review-Analysis\Data\Output\4Final_cons_list.csv"
df_proscons2.to_csv(output_file_proscons2, index=False)





