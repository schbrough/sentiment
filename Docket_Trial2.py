import requests
import csv
import time
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Constants (Replace with your actual API key)
API_KEY = "le3yia7EgtawuXKcCXUBSZqUBtSuGzyPyWtwmeUa"
BASE_URL = "https://api.regulations.gov/v4/comments"
HEADERS = {"X-Api-Key": API_KEY}

# Prompt user for docket number
docket_id = input("Enter the docket number: ").strip()
csv_filename = f"{docket_id}_comments.csv"

def get_comments(docket_id):
    """Fetch all comments from the given docket, handling pagination."""
    comments = []
    params = {"filter[docketId]": docket_id, "page[size]": 100}

    while True:
        response = requests.get(BASE_URL, headers=HEADERS, params=params)
        if response.status_code != 200:
            print(f"Error: {response.status_code}, {response.text}")
            break

        data = response.json()
        for item in data.get("data", []):
            comment_link = item.get("links", {}).get("self")
            comment_text = get_comment_text(comment_link) if comment_link else "No comment text"
            comments.append([item["id"], item["attributes"]["title"], comment_text])

        # Check for pagination
        next_page = data.get("links", {}).get("next")
        if not next_page:
            break
        params["page[token]"] = next_page.split("=")[-1]
        time.sleep(1)  # Avoid overwhelming the server

    return comments

def get_comment_text(comment_url):
    """Fetch the actual comment text from the detailed comment page."""
    response = requests.get(comment_url, headers=HEADERS)
    if response.status_code == 200:
        data = response.json()
        return data.get("data", {}).get("attributes", {}).get("comment", "No comment text")
    return "No comment text"

def save_to_csv(comments, filename):
    """Save comments to a CSV file."""
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Title", "Comment"])
        writer.writerows(comments)
    print(f"Comments saved to {filename}")

def analyze_sentiment(comment, sentiment_pipeline):
    """Analyze sentiment of a comment using a transformer model."""
    chunk_size = 512
    chunks = [comment[i:i+chunk_size] for i in range(0, len(comment), chunk_size)]

    sentiment_scores = {'1 star': 0, '2 stars': 0, '3 stars': 0, '4 stars': 0, '5 stars': 0}

    for chunk in chunks:
        result = sentiment_pipeline(chunk)[0]
        sentiment_scores[result['label']] += result['score']

    # Determine the sentiment with the highest score
    predominant_sentiment = max(sentiment_scores, key=sentiment_scores.get)
    return predominant_sentiment

def perform_sentiment_analysis(csv_filename):
    """Load CSV, filter comments, perform sentiment analysis, and generate a bar chart."""
    print("Performing sentiment analysis...")

    # Load CSV file
    df = pd.read_csv(csv_filename)

    # Filter out comments with fewer than 75 characters and those containing 'attached'
    filtered_df = df[(df['Comment'].str.len() >= 75) & (~df['Comment'].str.contains('attached', case=False))].copy()

    # Initialize sentiment analysis pipeline with a specific BERT model
    sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

    # Apply sentiment analysis
    filtered_df['Sentiment'] = filtered_df['Comment'].apply(lambda x: analyze_sentiment(x, sentiment_pipeline))

    # Map sentiment labels to more descriptive categories
    sentiment_mapping = {
        '1 star': 'Very Negative',
        '2 stars': 'Negative',
        '3 stars': 'Neutral',
        '4 stars': 'Positive',
        '5 stars': 'Very Positive'
    }
    filtered_df['Sentiment Category'] = filtered_df['Sentiment'].map(sentiment_mapping)

    # Generate a bar chart of the sentiment analysis results
    plt.figure(figsize=(10, 6))
    sns.countplot(data=filtered_df, x='Sentiment Category', palette='viridis', order=filtered_df['Sentiment Category'].value_counts().index)
    plt.title(f"Sentiment Distribution for Comments in Docket {docket_id}")
    plt.xlabel("Sentiment Category")
    plt.ylabel("Number of Comments")
    plt.show()

if __name__ == "__main__":
    # Get comments from regulations.gov
    comments = get_comments(docket_id)

    # Save comments to a CSV file
    save_to_csv(comments, csv_filename)

    # Perform sentiment analysis on the comments
    perform_sentiment_analysis(csv_filename)
