import pandas as pd
import re
import spacy
import string
import re
import nltk
nltk.download('all')

# Function to extract place and date
def extract_place_and_date(review):
    # Use regex to capture the place and date from the string
    match = re.search(r"Reviewed in (.+) on (\d{1,2} \w+ \d{4})", review)
    if match:
        place = match.group(1)
        date = pd.to_datetime(match.group(2), format='%d %B %Y').strftime('%d-%m-%Y')
        return pd.Series([place, date])

    else:
      review_split = review.split()
      place = review_split[2]
      date = pd.to_datetime(' '.join(review_split[-3:]), format='%d %B %Y').strftime('%d-%m-%Y')
      return pd.Series([place, date])
    
# Function to flag rows based on the number of words
def flag_for_analysis(review):
    # Split the review body into words and count them
    word_count = len(review.split())
    # Return True if more than 3 words, otherwise False
    return word_count > 3

# Function to map sentiment scores to labels
def map_sentiment(score):
    if score in [4, 5]:
        return 'POSITIVE'
    elif score in [1, 2]:
        return 'NEGATIVE'


file_path = r"C:\Users\tahiy\VS Code Scripts\Product-Review-Analysis\Data\ProductReviews_FinalDataset.xlsx"
df = pd.read_excel(file_path)

print("Data Preprocessing Initiated.")

# Add a row number starting from 1
df['review_id'] = range(1, len(df) + 1)

# Check number of rows with NaN values in the entire DataFrame
na_count_total = df.isna().sum().sum()
print(f"Total number of NaN values in the DataFrame: {na_count_total}")

# Check number of rows with NaN values in the 'review_body' column
na_count_reviewText = df['review_body'].isna().sum()
print(f"Number of rows with NaN in 'review_body' column: {na_count_reviewText}")

# Drop rows where the value in column 'B' is NaN
df = df.dropna(subset=['review_body'])

# Check for duplicates in the 'review_link' column
duplicates = df[df.duplicated(subset='review_link', keep=False)]

if not duplicates.empty:
    print(f"Found {len(duplicates)} duplicate rows.")

    # Remove duplicates, keeping only the first occurrence
    df = df.drop_duplicates(subset='review_link', keep='first')
    print(f"Removed duplicates. {len(df)} rows remaining.")
else:
    print("No duplicates found.")

print("Sentiment Rating Class Distribution: ", df['review_rating'].value_counts())

print("Extracting 'Place' and 'Date'.")
# Apply the function to the 'review_context' column
df[['Place', 'Date']] = df['review_context'].apply(extract_place_and_date)

print("Assigning sentiment labels to the dataset.")
# Apply the function to the 'sentiment_score' column
df['sentiment_label'] = df['review_rating'].apply(map_sentiment)

outputFilepath = r"C:\Users\tahiy\VS Code Scripts\Product-Review-Analysis\Data\Output\0Final_Preprocessed.csv"
print(f"Data Preproccessing Completed. Output dataset saved at {outputFilepath}")
df.to_csv(outputFilepath, index=False)


def clean_string(text):

    # Make lower
    text = text.lower()

    # Updated regex pattern to remove handles, URLs, emojis, and special characters, but keep numbers
    text = re.sub(r"([^A-Za-z0-9 \t])|(\w+:\/\/\S+)", ' ', text)

    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '<URL>', text)
  
    # Remove extra white spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove line breaks
    text = re.sub(r'\n', '', text)

    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # Remove stop words
    text = text.split()
    useless_words = ['just',
                        'actually',
                        'so',
                        'maybe',
                        'kind',
                        'of',
                        'pretty',
                        'seems',
                        'well',
                        'perhaps',
                        'almost',
                        'basically',
                        'literally',
                        'sort',
                        'of',
                        'somewhat',
                        'mostly',
                        'hardly',
                        'usually',
                        'apparently',
                        'often',
                        'probably',
                        'seemingly',
                        'hi', 'im', 'the', 'a', 'an', 'and', 'you', 'and', 'that', 'they', 'elsewhere', 'am', 'i', 'me', 'myself', 
                        'we',
                        'our',
                        'ours',
                        'ourselves',
                        'you',
                        'your',
                        'yours',
                        'yourself',
                        'yourselves',
                        'he',
                        'him',
                        'his',
                        'himself',
                        'she',
                        'her',
                        'hers',
                        'herself',
                        'it',
                        'its',
                        'itself',
                        'they',
                        'them',
                        'their',
                        'theirs',
                        'themselves',
                        'what',
                        'which',
                        'who',
                        'whom',
                        'this',
                        'that',
                        'these',
                        'those',
                        'am',
                        'is',
                        'are',
                        'was',
                        'were',
                        'be',
                        'been',
                        'being',
                        'have',
                        'has',
                        'had',
                        'having',
                        'do',
                        'does',
                        'did',
                        'doing',
                        'a',
                        'an',
                        'the',
                        'and',
                        'but',
                        'if',
                        'or',
                        'because',
                        'as',
                        'until',
                        'while',
                        'of',
                        'at',
                        'through',
                        'during',
                        'to',
                        'from',
                        'up',
                        'down',
                        'further',
                        'then',
                        'once',
                        'here',
                        'there',
                        'when',
                        'where',
                        'why',
                        'few',
                        'more',
                        'such',
                        'too',
                        'very',
                        's',
                        't',
                        'can',
                        'will',
                        'just',
                        'don',
                        'should',
                        'now'
                        ]

    text_filtered = [word for word in text if not word in useless_words]

    text_filtered = nlp(' '.join(text_filtered))

    return text_filtered

def lemmatize_text(text):

    # Process the text
    doc = nlp(text)

    # Extract the lemmas
    lemmas = [token.lemma_ for token in doc]

    # Join the lemmas into a single string
    lemmatized_text = ' '.join(lemmas)

    return lemmatized_text

print("Data Transformation Initiated.")

# Load spacy
nlp = spacy.load('en_core_web_sm')

print("Performing Review Text Cleaning.")
df['cleaned_review_body'] = df['review_body'].apply(clean_string)

print("Performing Lemmatization.")
df['lemma_review_body'] = df['cleaned_review_body'].apply(lemmatize_text)

print("Assessing the reviews eligibility for comprehensive analysis.")
# Apply the function to the 'review_body' column
df['forAnalysis'] = df['review_body'].apply(flag_for_analysis)

outputFilepath = r"C:\Users\tahiy\VS Code Scripts\Product-Review-Analysis\Data\Output\1Final_Transformed.csv"
print(f"Data Transformation Completed. Output dataset saved at {outputFilepath}")
df.to_csv(outputFilepath, index=False)
