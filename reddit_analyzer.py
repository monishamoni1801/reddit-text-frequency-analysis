# Install required packages
!pip install praw textblob wordcloud matplotlib numpy pandas plotly
!python -m textblob.download_corpora > /dev/null 2>&1

import praw
from textblob import TextBlob
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import pandas as pd
import re
import string
import warnings
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import logging

# Configure logging to suppress PRAW warnings
logging.getLogger('praw').setLevel(logging.CRITICAL)
logging.getLogger('prawcore').setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

# Set matplotlib style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'

class RedditAnalyzer:
    def __init__(self, subreddit_name='news'):
        # Initialize Reddit API
        self.reddit = praw.Reddit(
            client_id='jJgKlzwEQBzdXJWaFMv3xg',
            client_secret='eAYaacg-gkhc5xy8o6PyO7MD9z0RFQ',
            user_agent="script:reddit_analysis:v1.0"
        )
        self.subreddit = self.reddit.subreddit(subreddit_name)

        # Data structures (will be reset for each batch)
        self.word_counter = Counter()
        self.temporal_data = defaultdict(list)
        self.sentiment_data = defaultdict(list)
        self.start_time = datetime.now()

        # Comprehensive stop words list including function words
        self.stop_words = set([
            # Articles
            'a', 'an', 'the',

            # Auxiliary/helping verbs
            'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
            'do', 'does', 'did', 'have', 'has', 'had', 'having',
            'will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could',

            # Pronouns
            'i', 'me', 'my', 'mine', 'myself',
            'you', 'your', 'yours', 'yourself', 'yourselves',
            'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself',
            'we', 'us', 'our', 'ours', 'ourselves',
            'they', 'them', 'their', 'theirs', 'themselves',

            # Relative pronouns
            'who', 'whom', 'whose', 'which', 'that',

            # Prepositions
            'about', 'above', 'across', 'after', 'against', 'along', 'among', 'around',
            'at', 'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond',
            'by', 'down', 'during', 'for', 'from', 'in', 'inside', 'into', 'of', 'off',
            'on', 'onto', 'out', 'over', 'through', 'to', 'toward', 'under', 'until',
            'up', 'upon', 'with', 'within', 'without',

            # Conjunctions
            'and', 'but', 'or', 'nor', 'so', 'yet', 'for', 'although', 'because', 'since',
            'unless', 'until', 'while', 'where', 'after', 'before', 'though', 'if',

            # Common stop words
            'also', 'any', 'as', 'by', 'even', 'just', 'more', 'most', 'not', 'now',
            'only', 'other', 'some', 'such', 'than', 'then', 'there', 'these', 'this',
            'those', 'too', 'very', 'what', 'when', 'where', 'why', 'how', 'all', 'each',
            'every', 'many', 'much', 'few', 'little', 'own', 'same', 'so', 'again',
            'here', 'there', 'both', 'either', 'neither', 'no', 'none', 'never', 'ever',
            'however', 'therefore', 'thus', 'hence', 'meanwhile', 'otherwise', 'perhaps',
            'quite', 'rather', 'since', 'unless', 'until', 'whenever', 'wherever',
            'whether', 'while', 'why', 'within', 'without', 'yes', 'yet'
        ])

    def clean_text(self, text):
        """Clean text for analysis"""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'\w*\d\w*', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def analyze_sentiment(self, word, context):
        """Analyze sentiment of word in context"""
        blob = TextBlob(context)
        return blob.sentiment.polarity

    def is_valid_word(self, word):
        """Check if word should be counted - excludes stop words and short words"""
        return (len(word) > 2 and
                word not in self.stop_words and
                not word.startswith(('http', 'www')) and
                word.isalpha())

    def update_analysis(self, text):
        """Update analysis with new text"""
        cleaned = self.clean_text(text)
        words = re.findall(r'\b\w+\b', cleaned)

        for word in words:
            if self.is_valid_word(word):
                self.word_counter[word] += 1
                time_elapsed = (datetime.now() - self.start_time).total_seconds()
                self.temporal_data[word].append((time_elapsed, self.word_counter[word]))

                sentence = next((s for s in cleaned.split('.') if word in s), cleaned)
                sentiment = self.analyze_sentiment(word, sentence)
                self.sentiment_data[word].append(sentiment)

    def get_top_words(self, n=10):
        """Get top n words with counts and sentiment"""
        top_words = self.word_counter.most_common(n)
        return [(word, count, np.mean(self.sentiment_data.get(word, [0])))
                for word, count in top_words]

    def create_wordcloud(self):
        """Generate wordcloud visualization"""
        try:
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=50
            ).generate_from_frequencies(self.word_counter)

            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Word Frequency Cloud (Current Batch)', pad=20)
            plt.show()
        except Exception as e:
            print(f"Couldn't generate wordcloud: {str(e)}")

    def create_bar_chart(self, top_n=15):
        """Create interactive bar chart"""
        try:
            top_words = self.get_top_words(top_n)
            df = pd.DataFrame(top_words, columns=['Word', 'Frequency', 'Sentiment'])

            fig = px.bar(df,
                        x='Frequency',
                        y='Word',
                        color='Sentiment',
                        color_continuous_scale='RdYlGn',
                        title=f'Top {top_n} Words in Current Batch',
                        labels={'Frequency': 'Count', 'Word': ''},
                        hover_data={'Sentiment': ':.2f'})

            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=600
            )
            fig.show()
        except Exception as e:
            print(f"Couldn't generate bar chart: {str(e)}")

    def reset_counters(self):
        """Reset all counters for a new batch"""
        self.word_counter = Counter()
        self.temporal_data = defaultdict(list)
        self.sentiment_data = defaultdict(list)
        self.start_time = datetime.now()

    def run_analysis(self, batch_size=50):
        """Main analysis loop with independent batches"""
        print("Starting Reddit analysis... Press Ctrl+C to stop")
        print(f"Each visualization will show results from an independent batch of ~{batch_size} unique words\n")

        try:
            total_comments = 0
            batch_number = 1

            while True:
                # Reset counters for new batch
                self.reset_counters()
                batch_comments = 0

                print(f"\nStarting batch {batch_number}...")

                for comment in self.subreddit.stream.comments(skip_existing=True):
                    self.update_analysis(comment.body)
                    total_comments += 1
                    batch_comments += 1

                    # Print progress
                    if batch_comments % 10 == 0:
                        top_word = self.word_counter.most_common(1)[0][0] if self.word_counter else "N/A"
                        print(f"\rBatch {batch_number}: {len(self.word_counter)} unique words from {batch_comments} comments | Top: '{top_word}'", end="", flush=True)

                    # Check if we've collected enough unique words for this batch
                    if len(self.word_counter) >= batch_size:
                        print(f"\n\nCompleted batch {batch_number} with {len(self.word_counter)} unique words from {batch_comments} comments")
                        print(f"Total comments processed: {total_comments}")

                        # Generate visualizations for this batch
                        self.create_wordcloud()
                        self.create_bar_chart()

                        # Prepare for next batch
                        batch_number += 1
                        break

        except KeyboardInterrupt:
            print("\n\nAnalysis stopped by user")
            print(f"Processed {total_comments} total comments across {batch_number-1} complete batches")
            if len(self.word_counter) > 0:

                print(f"\nFinal incomplete batch had {len(self.word_counter)} unique words from {batch_comments} comments")
                print("Generating visualizations for final partial batch...")
                self.create_wordcloud()
                self.create_bar_chart()

# Run in Colab
if __name__ == "__main__":
    analyzer = RedditAnalyzer('news')
    analyzer.run_analysis(batch_size=50)
