
📊 Reddit Text Frequency Analysis

This project performs real-time text frequency and sentiment analysis on Reddit comments. By streaming live data from any subreddit, it processes the text to extract single words, analyzes their frequency of occurrence, determines their sentiment polarity, and generates engaging visualizations like word clouds and interactive bar charts.

The goal is to provide insights into trending words, overall mood, and text usage patterns within online discussions.

🚀 How It Works

1. Reddit API Integration (PRAW)
      The project uses the PRAW (Python Reddit API Wrapper) to stream comments from a chosen subreddit (default: r/news).
      
      Comments are collected continuously in real time.
2. Text Preprocessing & Cleaning

      Converts all text to lowercase.
      
      Removes links, numbers, punctuation, and special characters.
      
      Excludes stop words (common words like “the”, “and”, “is”) for better accuracy.

3. Single-Word Extraction & Filtering

      Splits the text into words.
      
      Only considers valid words (alphabetic, length > 2, not in stopwords).

4. Frequency Analysis

      Each valid word is counted.
      
      Tracks how frequently words appear over time in batches (e.g., 50 unique words per batch).

5. Sentiment Analysis (TextBlob)

      Each word’s context sentence is analyzed.
      
      Sentiment polarity is measured on a scale of -1 (negative) to +1 (positive).

6. Visualizations

      Word Cloud: Shows the most frequent words, where size = frequency.
      
      Interactive Bar Chart (Plotly):
      
      Displays top words and their frequency.
      
      Colors indicate average sentiment (red = negative, green = positive).

7. Batch-Based Analysis

      Data resets after each batch, ensuring fresh insights every cycle.
      
      Each batch works independently, allowing comparisons across time windows.
