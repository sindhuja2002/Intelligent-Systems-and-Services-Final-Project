import nltk
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import sys
import csv
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import string


# Download necessary NLTK datasets
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# Increase field size limit
#csv.field_size_limit(sys.maxint)

maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

# Define chunk size and sampling ratio
CHUNK_SIZE = 100000  # Read 100,000 rows at a time
SAMPLE_RATIO = 0.10  # Extract 10% of total data

sampled_chunks = []

for chunk in pd.read_csv("data/news.csv/news.csv", usecols=["content"], dtype=str, encoding="utf-8",
                         on_bad_lines="skip", low_memory=True, chunksize=CHUNK_SIZE, engine="python"):
    chunk_sample = chunk.sample(frac=SAMPLE_RATIO, random_state=42)  # Sample 10% of each chunk
    sampled_chunks.append(chunk_sample)

# Combine all sampled chunks
df_sampled = pd.concat(sampled_chunks, ignore_index=True)

print(f" Final Sampled Dataset Size: {len(df_sampled)} rows") 

# Initialize stopwords and stemmer

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# 🔹 Function to Compute Vocabulary Size in Chunks
def get_vocabulary_size(text_series, chunk_size=5000):
    vocab = Counter()  # Counter to store word frequencies
    for i in range(0, len(text_series), chunk_size):
        chunk_tokens = []
        
        # Tokenize each text in the chunk
        for text in text_series[i:i + chunk_size]:
            if text:  # Check if the text is not None or empty
                tokens = word_tokenize(text)  # Tokenize each text
                tokens = [token.lower() for token in tokens if token.isalpha()]  # Remove non-alphabetic tokens and convert to lowercase
                chunk_tokens.extend(tokens)  # Add tokens to the chunk_tokens list
        
        # Update vocabulary with tokens from this chunk
        vocab.update(chunk_tokens)
    
    return len(vocab)  # Return the vocabulary size

# Compute initial vocabulary size (before processing)
vocab_size_before = get_vocabulary_size(df_sampled['content'])

print(f" Initial Vocabulary Size: {vocab_size_before}")

# 🔹 Remove null values from the 'content' column
df_sampled = df_sampled.dropna(subset=['content'])

# 🔹 Function to Preprocess Text: Remove stopwords and apply stemming
def preprocess_text(text):
    tokens = word_tokenize(text)  # Tokenize the text
    tokens = [token.lower() for token in tokens if token.isalpha()]  # Remove non-alphabetic tokens and convert to lowercase
    tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    tokens = [stemmer.stem(token) for token in tokens]  # Apply stemming
    return tokens

# 🔹 Function to Compute Vocabulary Size After Preprocessing
def get_vocabulary_size_processed(text_series, chunk_size=5000):
    vocab = Counter()
    for i in range(0, len(text_series), chunk_size):
        chunk_tokens = [preprocess_text(text) for text in text_series[i:i + chunk_size]]
        for tokens in chunk_tokens:
            vocab.update(tokens)
    return len(vocab)

# Compute vocabulary size after preprocessing (removing stopwords and stemming)
vocab_size_after = get_vocabulary_size_processed(df_sampled['content'])

print(f" Vocabulary Size After Preprocessing: {vocab_size_after}")

import re

# Function to count URLs
def count_urls(text):
    return len(re.findall(r'http[s]?://\S+', text))

# Count URLs in the dataset
df_sampled['url_count'] = df_sampled['content'].apply(count_urls)

# Check some stats about URL counts
print(df_sampled['url_count'].describe())

# Function to count dates
def count_dates(text):
    return len(re.findall(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', text))

# Count dates in the dataset
df_sampled['date_count'] = df_sampled['content'].apply(count_dates)

# Check some stats about date counts
print(df_sampled['date_count'].describe())

# Function to count numeric values
def count_numbers(text):
    return len(re.findall(r'\b\d+\b', text))

# Count numbers in the dataset
df_sampled['num_count'] = df_sampled['content'].apply(count_numbers)

# Check some stats about numeric counts
print(df_sampled['num_count'].describe())

total_urls = df_sampled["url_count"].sum()
total_dates = df_sampled["date_count"].sum()
total_numbers = df_sampled["num_count"].sum()

print(f"Total URLs: {total_urls}")
print(f"Total Dates: {total_dates}")
print(f"Total Numbers: {total_numbers}")

# Initialize an empty frequency distribution
freq_dist_before = FreqDist()

# Define chunk size
chunk_size = 10000  # Adjust based on memory constraints

# Process in chunks
for i in range(0, len(df_sampled), chunk_size):
    chunk = df_sampled["content"][i : i + chunk_size].dropna()  # Avoid null values
    chunk_tokens = [word_tokenize(text.lower()) for text in chunk]  # Tokenize each text
    chunk_tokens = [item for sublist in chunk_tokens for item in sublist]  # Flatten list
    freq_dist_before.update(chunk_tokens)  # Update frequency distribution

# Get the 100 most frequent words
most_frequent_words = freq_dist_before.most_common(100)

print(most_frequent_words)

# Load English stopwords and punctuation
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

# Initialize frequency distribution and stemmer
freq_dist = FreqDist()
stemmer = PorterStemmer()

# Define chunk size
chunk_size = 10000  

# Process dataset in chunks
for i in range(0, len(df_sampled), chunk_size):
    chunk = df_sampled["content"][i : i + chunk_size].dropna()  # Drop null values in the chunk
    chunk_tokens = [word_tokenize(text.lower()) for text in chunk]  # Tokenize each text
    
    # Flatten the list of tokens
    chunk_tokens = [item for sublist in chunk_tokens for item in sublist]  
    
    # Filter out stopwords, punctuation, and non-alphabetic tokens
    filtered_tokens = [
        stemmer.stem(word)  # Apply stemming
        for word in chunk_tokens
        if word.isalpha() and word not in stop_words and word not in punctuation
    ]
    
    # Update frequency distribution
    freq_dist.update(filtered_tokens)  

# Get the 100 most frequent words after preprocessing
most_frequent_words = freq_dist.most_common(100)

# Print the 100 most frequent words
print(most_frequent_words)



# Function to safely plot word frequencies
def plot_word_frequencies(freq_dist, title):
    if not freq_dist or len(freq_dist) == 0:
        print(f"Warning: No words available for {title}")
        return
    
    # Get the top 30 most common words
    most_common_words = freq_dist.most_common(30)
    words, counts = zip(*most_common_words)  # Unzip words and their counts
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.bar(words, counts, color='skyblue')
    plt.xticks(rotation=45, ha='right')  # Rotate labels for better readability
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add a light grid
    plt.show()

# Plotting the 30 most frequent words before preprocessing
plot_word_frequencies(freq_dist_before, "Top 30 Frequent Words (Before Preprocessing)")

# Plotting the 30 most frequent words after preprocessing
plot_word_frequencies(freq_dist, "Top 30 Frequent Words (After Preprocessing)")