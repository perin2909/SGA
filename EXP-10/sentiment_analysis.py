import nltk
nltk.download('vader_lexicon')
import pandas as pd
import re
from nltk.sentiment import SentimentIntensityAnalyzer
# 1. Load the dataset
# The actual headers are in the second row (index 1) of the CSV
# Based on the previous df.head() output, pd.read_csv(header=1) resulted in 'Unnamed: X' columns,
# and the actual headers ('id', 'tweetText') were loaded as the first row of data (df.iloc[0]).
# We will load with header=1 as originally intended, but then manually correct the column names.
df = pd.read_csv("tweets.csv",)   # make sure tweets.csv is in the same folder
# Correct column names: use the first row of the loaded DataFrame as column headers
df.columns = df.iloc[0]
# Drop the row that was just used as headers
df = df[1:].reset_index(drop=True)
print("Sample data after correcting headers:")
print(df.head())
# 2. Basic preprocessing function
def clean_tweet(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+", "", text)                    # remove mentions
    text = re.sub(r"#", "", text)                       # remove hashtag symbol
    text = re.sub(r"[^a-zA-Z\s]", "", text)             # keep only letters and spaces
    text = re.sub(r"\s+", " ", text).strip()            # remove extra spaces
    return text
# Apply cleaning to the 'tweetText' column
df["clean_tweet"] = df["tweetText"].apply(clean_tweet)
print("\nCleaned tweets:")
print(df[["tweetText", "clean_tweet"]].head())
# 3. Initialize VADER
sia = SentimentIntensityAnalyzer()
# 4. Function to get sentiment label from compound score
def get_sentiment_label(score, pos_thresh=0.05, neg_thresh=-0.05):
    if score >= pos_thresh:
        return "positive"
    elif score <= neg_thresh:
        return "negative"
    else:
        return "neutral"
# 5. Apply sentiment analysis
compound_scores = []
sentiment_labels = []
for t in df["clean_tweet"]:
    scores = sia.polarity_scores(t)
    #print(scores)
    compound = scores["compound"]
    compound_scores.append(compound)
    sentiment_labels.append(get_sentiment_label(compound))
df["compound_score"] = compound_scores
df["predicted_sentiment"] = sentiment_labels
print("\nSentiment results:")
print(df[["tweetText", "clean_tweet", "compound_score", "predicted_sentiment"]])
# 6. (Optional) Evaluate accuracy if true labels are available
if "label" in df.columns:
    correct = (df["label"].str.lower() == df["predicted_sentiment"].str.lower()).sum()
    total = len(df)
    accuracy = correct / total * 100
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
# 7. Save results to a new CSV
df.to_csv("tweets_with_sentiment.csv", index=False)
print("\nResults saved to tweets_with_sentiment.csv")