import praw
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
from datetime import datetime, timezone
import tweepy
import yfinance as yf

#API Keys
TWITTER_API_KEY = "Insert Twitter API"
TWITTER_API_SECRET_KEY = "Insert Twitter API Secret"
TWITTER_BEARER_TOKEN = "Insert Twitter Token"
TWITTER_ACCESS_TOKEN = "Insert Twitter Token"
TWITTER_ACCESS_TOKEN_SECRET = "Insert Twitter Token"

REDDIT_CLIENT_ID = "Insert Reddit Token"
REDDIT_CLIENT_SECRET = "Insert Reddit Token"
NEWS_API_KEY = "Insert Google News API Token"


#Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

#Date range for the analysis
START_DATE = "2024-10-01T00:00:00Z"
END_DATE = "2024-10-25T23:59:59Z"

#Dates for Twitter API
TWITTER_START_DATE = datetime(2024, 10, 2, 21, 0, 0).isoformat() + "Z"
TWITTER_END_DATE = datetime(2024, 10, 25, 23, 59, 59).isoformat() + "Z"

#Dates for News API
NEWS_API_START_DATE = "2024-10-01T00:00:00Z"

#Twitter API Setup (Tweepy)
def get_twitter_data():
    client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
    query = "Donald Trump OR Kamala Harris"
    try:
        tweets = client.search_recent_tweets(query=query, start_time=TWITTER_START_DATE, end_time=TWITTER_END_DATE,
                                             tweet_fields=["created_at", "text"], max_results=20)
        twitter_data = [tweet.text for tweet in tweets.data] if tweets.data else []
        return twitter_data
    except tweepy.TooManyRequests as e:
        print("Rate limit exceeded on Twitter. Skipping Twitter data for now.")
        return []

#Reddit API Setup (PRAW)
def get_reddit_data():
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent="sentiment_analysis_project_by_Username"
    )

    reddit_data = []
    for submission in reddit.subreddit("politics").top(limit=100):
        created_date = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).strftime('%Y-%m-%d')
        if (START_DATE[:10] <= created_date <= END_DATE[:10]) and ("Donald Trump" in submission.title or "Kamala Harris" in submission.title):
            reddit_data.append(submission.title + " " + submission.selftext)
    return reddit_data

#Google News API Setup (NewsAPI)
def get_google_news_data():
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    articles = newsapi.get_everything(
        q="Donald Trump OR Kamala Harris",
        from_param=NEWS_API_START_DATE[:10],
        to=END_DATE[:10],
        language='en'
    )
    google_news_data = [article['title'] for article in articles['articles']]
    return google_news_data

#Sentiment Analysis with VADER
def analyze_sentiment(text):
    sentiment = analyzer.polarity_scores(text)
    if sentiment["compound"] >= 0.2:
        return "positive"
    elif sentiment["compound"] <= -0.2:
        return "negative"
    else:
        return "neutral"
    
#Candidate Stock Associations
def identify_benefit_stocks():
    sector_keywords = {
        "oil": ["XOM", "CVX"],
        "defense": ["LMT", "RTX", "HON"],
        "finance": ["GS", "JPM"],
        "renewables": ["TSLA", "ENPH"],
        "tech": ["AAPL", "NVDA", "TSLA"],
        "crypto": ["COIN"]
    }

    trump_keywords = ["oil", "defense", "finance", "crypto", "tech"]
    harris_keywords = ["renewables", "tech", "defense"]

    trump_stocks = [stock for sector in trump_keywords for stock in sector_keywords[sector]]
    harris_stocks = [stock for sector in harris_keywords for stock in sector_keywords[sector]]

    return trump_stocks, harris_stocks

#Stock Performance in the month leading up to Election Results
def get_stock_performance(tickers, start_date, end_date):
    performance = {}
    for ticker in tickers:
        data = yf.Ticker(ticker).history(start=start_date, end=end_date)
        if not data.empty:
            growth = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
            performance[ticker] = growth
    return performance

#Main
def main():
    twitter_data = get_twitter_data()
    reddit_data = get_reddit_data()
    news_data = get_google_news_data()

    all_data = twitter_data + reddit_data + news_data

    sentiment_scores = [analyze_sentiment(text) for text in all_data]

    positive = sentiment_scores.count("positive")
    negative = sentiment_scores.count("negative")

    print(f"Positive Sentiment: {positive}")
    print(f"Negative Sentiment: {negative}")

    trump_stocks, harris_stocks = identify_benefit_stocks()

    if positive > negative:
        print("The likely winner based on sentiment is: Kamala Harris")
        winning_stocks = harris_stocks
    else:
        print("The likely winner based on sentiment is: Donald Trump")
        winning_stocks = trump_stocks

    #Analyze Stock Performance
    start_date = "2024-10-01"
    end_date = "2024-10-25"
    stock_performance = get_stock_performance(winning_stocks, start_date, end_date)

    print("Investment Opportunities:")
    for stock, growth in stock_performance.items():
        print(f"{stock}: {growth:.2f}% potential growth")

if __name__ == "__main__":
    main()