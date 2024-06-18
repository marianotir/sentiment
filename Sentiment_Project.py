
# #########################
# Import libraries 
# #########################

import os
import yfinance as yf
import pandas as pd
import finnhub
from datetime import datetime
from datetime import timedelta
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.stats import pearsonr
import numpy as np
import re
import pandas_ta as ta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from math import ceil

import plotly.graph_objects as go


# #########################
# Define functions
# #########################

# --------------------------------------------------
#  Tickers collection functions 
# --------------------------------------------------

# Sp500 tickers
def get_spx500_stocks():
    """
    Get the list of stocks from wikipedia
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    stocks = pd.read_html(url, header=0)[0]
    return stocks['Symbol'].tolist()

# Russell 3000 tickers
def get_russell3000_tickers():

    try:
        url = "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund"
        
        # Tickers are in the first column of the CSV file
        holdings_df = pd.read_csv(url, skiprows=10)  
        
        # Assume tickers are in the first column, get all rows where the first column is not empty
        tickers = holdings_df.iloc[:, 0].dropna().tolist()

        # Filter tickers: only keep strings with uppercase letters and no special characters
        valid_tickers = [ticker for ticker in tickers if re.match(r'^[A-Z]+$', ticker)]
        
        return valid_tickers
    
    except Exception as e:
        print(f"Error fetching Russell 3000 tickers: {e}")
        return []


# --------------------------------------------------
#  Price data collection functions 
# --------------------------------------------------

def get_stocks_data_hist(tickers, period, interval):

    all_data = []
    
    for ticker in tickers:
        data = yf.download(ticker, period=period, interval=interval)
        data['Ticker'] = ticker
        all_data.append(data)
    
    # Concatenate all dataframes
    combined_data = pd.concat(all_data)
    
    # Reset index to have 'Date' as a column
    combined_data.reset_index(inplace=True)
    
    return combined_data


def get_sxp500_stocks_data_hist():

    # Get the list of stocks
    stocks = get_spx500_stocks()

    # Get the data
    df = get_stocks_data_hist(stocks,'1y','1d')

    return df


def get_sxp500_stocks_data_hist_1h():

    # Get the list of stocks
    stocks = get_spx500_stocks()

    # Get the data
    df = get_stocks_data_hist(stocks,'3mo','1h')

    return df


def get_stocks_data_hist_1h(stocks):

    # Get the data
    df = get_stocks_data_hist(stocks,'3mo','1h')

    return df


# --------------------------------------------------
#  News collection functions 
# --------------------------------------------------

def get_news_by_stock(client, symbol, from_date, to_date):
    """
    Get the stock news for a specific symbol from Finnhub.
    """
    while True:
        try:
            news_data = client.company_news(symbol=symbol, _from=from_date, to=to_date)
            break
        except finnhub.FinnhubAPIException as e:
            if e.status_code == 429:
                print("API limit reached. Waiting for 31 seconds.")
                time.sleep(40)
            else:
                raise e

    news_df = pd.DataFrame(news_data)
    
    if 'datetime' in news_df.columns:
        # Convert Unix timestamps to human-readable datetime
        news_df['datetime'] = news_df['datetime'].apply(lambda x: datetime.fromtimestamp(x))
    else:
        print(f"No data for the period selected: {from_date} to {to_date}")
    
    return news_df


def download_news_in_chunks(client, symbol):
    """
    Download news in chunks for the last year, day by day.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    current_date = start_date
    all_news = []

    while current_date < end_date:
        next_date = current_date + timedelta(days=1)
        from_date = current_date.strftime('%Y-%m-%d')
        to_date = next_date.strftime('%Y-%m-%d')
        
        print(f"Downloading news from {from_date} to {to_date}")
        
        news_df = get_news_by_stock(client, symbol, from_date, to_date)
        if not news_df.empty:
            all_news.append(news_df)
        
        current_date = next_date

    return pd.concat(all_news, ignore_index=True) if all_news else pd.DataFrame()


def get_all_news(stocks, finnhub_client):

    # Create data folder if it does not exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # Get the news

    for stock in stocks:
        print("-" * 50)
        print(f"Downloading news for {stock}")
        print("-" * 50)

        # Check if the news file already exists
        if os.path.exists(f'data/news_{stock}.csv'):
            print(f"News for {stock} already downloaded")
            continue

        # Download the news
        df_all_news_stock = download_news_in_chunks(finnhub_client, stock)

        # Save the news to a CSV file
        df_all_news_stock.to_csv(f'data/news_{stock}.csv', index=False)

    print('All news downloaded')


# --------------------------------------------------
#  Sentiment analysis functions 
# --------------------------------------------------

# ------------ Sentiment score functions ------------
def analyze_sentiment(news_df):
    """
    Perform sentiment analysis on the news headlines and summaries.
    """
    analyzer = SentimentIntensityAnalyzer()
    
    df = news_df.copy()

    def sentiment_score(text):
        if pd.isnull(text):
            return 0  # Return neutral score for missing text
        score = analyzer.polarity_scores(text)['compound']
        # Return the score as is on a scale of -1 to 1
        return score
    # news_df = df_news.copy()
    df['headline_sentiment'] = df['headline'].apply(sentiment_score)
    df['summary_sentiment'] = df['summary'].apply(sentiment_score)
    
    return df

# ------------ Sentiment aggregation functions ------------

def extreme_sentiment(group, sentiment_col):
    max_sentiment = group[sentiment_col].max()
    min_sentiment = group[sentiment_col].min()
    return max_sentiment if abs(max_sentiment) > abs(min_sentiment) else min_sentiment


def weighted_avg(group, sentiment_col):
    weights = group[sentiment_col].abs()
    weighted_sentiment = (group[sentiment_col] * weights).sum() / weights.sum()
    return weighted_sentiment


def aggregate_average_sentiment(news_df, freq='H'):

    # Make a copy of the DataFrame
    df = news_df.copy()

    # Convert 'datetime' to datetime type if not already
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

    # Drop rows with invalid datetime values
    df = df.dropna(subset=['datetime'])

    # Ensure DataFrame is sorted by datetime
    df = df.sort_values('datetime')

    # Set 'datetime' as the index
    df.set_index('datetime', inplace=True)

    # Resample the data
    average_sentiment = df.resample(freq).agg({
        'headline_sentiment': 'mean',
        'summary_sentiment': 'mean'
    }).rename(columns={
        'headline_sentiment': 'avg_headline_sentiment',
        'summary_sentiment': 'avg_summary_sentiment'
    }).reset_index()

    # Fill missing values with 0
    average_sentiment.fillna(0, inplace=True)
    
    return average_sentiment


def aggregate_sentiment_weighted(news_df, freq='H'):
    df = news_df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    weighted_sentiment = df.resample(freq).apply(
        lambda x: pd.Series({
            'weighted_headline_sentiment': weighted_avg(x, 'headline_sentiment'),
            'weighted_summary_sentiment': weighted_avg(x, 'summary_sentiment')
        })
    ).reset_index()

    # fill missing values with 0
    weighted_sentiment.fillna(0, inplace=True)
    
    return weighted_sentiment


def aggregate_sum_sentiment(news_df, freq='H'):
    df = news_df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    sum_sentiment = df.resample(freq).agg({
        'headline_sentiment': 'sum',
        'summary_sentiment': 'sum'
    }).rename(columns={
        'headline_sentiment': 'sum_headline_sentiment',
        'summary_sentiment': 'sum_summary_sentiment'
    }).reset_index()

    # fill missing values with 0
    sum_sentiment.fillna(0, inplace=True)
    
    return sum_sentiment


def sentiment_extreme(news_df, freq='H'):
    """
    Aggregate sentiment by considering the most extreme sentiment score for each time unit.
    """
    df = news_df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    extreme_sentiment_df = df.resample(freq).apply(
        lambda x: pd.Series({
            'extreme_headline_sentiment': extreme_sentiment(x, 'headline_sentiment'),
            'extreme_summary_sentiment': extreme_sentiment(x, 'summary_sentiment')
        })
    ).reset_index()

    # fill missing values with 0
    extreme_sentiment_df.fillna(0, inplace=True)
    
    return extreme_sentiment_df


def classify_sentiment(row, positive_threshold=0.1, negative_threshold=-0.1):
    """
    Classify the sentiment of a news article based on both headline and summary sentiments.
    """
    avg_sentiment = (row['headline_sentiment'] + row['summary_sentiment']) / 2
    if avg_sentiment > positive_threshold:
        return 'positive'
    elif avg_sentiment < negative_threshold:
        return 'negative'
    else:
        return 'neutral'


def classify_and_aggregate_sentiment(df_sentiment, freq='H', positive_threshold=0.1, negative_threshold=-0.1, window_size=4):
    """
    Classify news articles by sentiment and aggregate to specified frequency with rolling window.
    """
    # Classify articles based on sentiment scores
    df = df_sentiment.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df['sentiment_class'] = df.apply(classify_sentiment, positive_threshold=positive_threshold, negative_threshold=negative_threshold, axis=1)

    # Resample to specified frequency and count the number of positive, negative, and neutral articles
    df_resampled = df.resample(freq).apply(lambda x: pd.Series({
        'positive': (x['sentiment_class'] == 'positive').sum(),
        'negative': (x['sentiment_class'] == 'negative').sum(),
        'neutral': (x['sentiment_class'] == 'neutral').sum()
    })).fillna(0)

    # Apply a rolling window to calculate the cumulative number of articles over the past window_size periods
    df_resampled['positive_rolling'] = df_resampled['positive'].rolling(window=window_size, min_periods=1).sum()
    df_resampled['negative_rolling'] = df_resampled['negative'].rolling(window=window_size, min_periods=1).sum()
    df_resampled['neutral_rolling'] = df_resampled['neutral'].rolling(window=window_size, min_periods=1).sum()

    return df_resampled.reset_index()


# --------------------------------------------------
# Trading Indicator score functions
# --------------------------------------------------

# Function to calculate rolling Pearson correlation and p-value
def rolling_pearsonr(x, y, window):
    corr = []
    pval = []
    for i in range(len(x)):
        if i < window - 1:
            corr.append(np.nan)
            pval.append(np.nan)
        else:
            x_slice = x[i-window+1:i+1]
            y_slice = y[i-window+1:i+1]
            mask = x_slice.notna() & y_slice.notna()
            x_slice = x_slice[mask]
            y_slice = y_slice[mask]
            if len(x_slice) > 1 and len(y_slice) > 1:
                r, p = pearsonr(x_slice, y_slice)
                corr.append(r)
                pval.append(p)
            else:
                corr.append(np.nan)
                pval.append(np.nan)
    return np.array(corr), np.array(pval)

# Momentum indicator
def calculate_roc(df, column='sentiment_score', window=12):
    df[f'roc_{column}'] = df[column].diff(window) / df[column].shift(window)
    df[f'roc_{column}'].fillna(0, inplace=True)
    return df

# Trend Saturation indicator
def calculate_rsi(df, column='sentiment_score', window=14):
    delta = df[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df[f'rsi_{column}'] = 100 - (100 / (1 + rs))
    df[f'rsi_{column}'].fillna(0, inplace=True)
    return df

# Trend indicator
def calculate_moving_average(df, column='sentiment_score', window=5):
    df[f'sma_{column}'] = df[column].rolling(window=window).mean()
    df[f'sma_{column}'].fillna(0, inplace=True)
    return df

# Trend indicator
def calculate_slope(df, column='sma_sentiment_score', window=5):
    def slope(series):
        y = series.values
        x = range(len(y))
        if len(y) == window:
            slope = np.polyfit(x, y, 1)[0]
            return slope
        else:
            return np.nan
    df[f'slope_{column}'] = df[column].rolling(window=window).apply(slope, raw=False)
    df[f'slope_{column}'].fillna(0, inplace=True)
    return df



# ----------- Trading indicators news volumne indicators -----------

def calculate_count_indicators(df, columns, window_roc=12, window_rsi=14, window_ma=5, window_slope=5):

    for column in columns:
        df = calculate_roc(df, column=column, window=window_roc)
        df = calculate_rsi(df, column=column, window=window_rsi)
        df = calculate_moving_average(df, column=column, window=window_ma)
        df = calculate_slope(df, column=f'sma_{column}', window=window_slope)
    return df


# --------------------------------------------------
#  Trading signals functions 
# --------------------------------------------------

def generate_rsi_signals(df, sentiment_col, positive_threshold=0.1, negative_threshold=-0.1, price_rsi_low=40, price_rsi_high=60, sentiment_rsi_low=40, sentiment_rsi_high=60):
    sentiment_rsi_col = f'rsi_{sentiment_col}'
    signal_col = f'signal_rsi_{sentiment_col}'

    # initialize the signal column
    df[signal_col] = 0

    df.loc[(df['rsi'] < price_rsi_low) & (df[sentiment_rsi_col] > sentiment_rsi_high) & (df[sentiment_col] > positive_threshold), signal_col] = 'buy'
    df.loc[(df['rsi'] > price_rsi_high) & (df[sentiment_rsi_col] < sentiment_rsi_low) & (df[sentiment_col] < negative_threshold), signal_col] = 'sell'
    
    return df


def generate_sma_signals(df, sentiment_col, fast_window=5, slow_window=20, positive_threshold=0.1, negative_threshold=-0.1):
    # Calculate fast and slow SMAs for the price
    df['fast_sma'] = df['Close'].rolling(window=fast_window).mean()
    df['slow_sma'] = df['Close'].rolling(window=slow_window).mean()
    
    # Generate buy and sell signals
    signal_col = f'signal_sma_{sentiment_col}'
    df[ f'signal_sma_{sentiment_col}'] = 0
    
    df.loc[(df['fast_sma'] > df['slow_sma']) & 
           (df['fast_sma'].shift(1) <= df['slow_sma'].shift(1)) & 
           (df[f'sma_{sentiment_col}'] > positive_threshold), signal_col] = 'buy'
    
    df.loc[(df['fast_sma'] < df['slow_sma']) & 
           (df['fast_sma'].shift(1) >= df['slow_sma'].shift(1)) & 
           (df[f'sma_{sentiment_col}'] < negative_threshold), signal_col] = 'sell'
    
    # Replace '' and NaN values with 0
    df[signal_col] = df[signal_col].replace({'': 0}).fillna(0)

    # Convert 'buy' and 'sell' to 1 and -1
    df[signal_col] = df[signal_col].replace({'buy': 1, 'sell': -1})
    
    return df


def generate_roc_signals(df, sentiment_col, positive_threshold=0.1, negative_threshold=-0.1):
    sentiment_roc_col = f'roc_{sentiment_col}'
    signal_col = f'signal_roc_{sentiment_col}'

    # Initialize the signal column
    df[signal_col] = 0

    # Generate buy signal: Detect turning points in ROC
    buy_condition = (
        (df['roc'] > 0) &
        (df['roc'].shift(2) <= 0) &
        (df[sentiment_roc_col] > 0) &
        (df[sentiment_col] > positive_threshold)
    )
    df.loc[buy_condition, signal_col] = 'buy'
    
    # Generate sell signal: Detect turning points in ROC
    sell_condition = (
        (df['roc'] < 0) &
        (df['roc'].shift(2) >= 0) &
        (df[sentiment_roc_col] < 0) &
        (df[sentiment_col] < negative_threshold)
    )
    df.loc[sell_condition, signal_col] = 'sell'

    # Replace '' and NaN values with 0
    df[signal_col] = df[signal_col].replace({'': 0}).fillna(0)

    # Convert 'buy' and 'sell' to 1 and -1
    df[signal_col] = df[signal_col].replace({'buy': 1, 'sell': -1})
    
    return df


def generate_slope_signals(df, sentiment_col, positive_threshold=0.1, negative_threshold=-0.1):
    sentiment_slope_col = f'slope_{sentiment_col}'
    signal_col = f'signal_slope_{sentiment_col}'

    # initialize the signal column
    df[signal_col] = 0

    # Generate buy signal: Detect turning points in slope
    buy_condition = (
        (df['slope'] > 0) &
        (df['slope'].shift(2) <= 0) &
        (df[sentiment_slope_col] > 0) &
        (df[sentiment_col] > positive_threshold)
    )
    df.loc[buy_condition, signal_col] = 'buy'
    
    # Generate sell signal: Detect turning points in slope
    sell_condition = (
        (df['slope'] < 0) &
        (df['slope'].shift(2) >= 0) &
        (df[sentiment_slope_col] < 0) &
        (df[sentiment_col] < negative_threshold)
    )
    df.loc[sell_condition, signal_col] = 'sell'

    # Replace '' and NaN values with 0
    df[signal_col] = df[signal_col].replace({'': 0}).fillna(0)

    # Convert 'buy' and 'sell' to 1 and -1
    df[signal_col] = df[signal_col].replace({'buy': 1, 'sell': -1})
    
    return df


def generate_sentiment_volume_signals(df, 
                                      price_rsi_col='rsi', 
                                      price_rsi_low=40, 
                                      price_rsi_high=60, 
                                      slope_positive_col='slope_positive_rolling', 
                                      sma_positive_col='sma_positive_rolling', 
                                      slope_negative_col='slope_negative_rolling', 
                                      sma_negative_col='sma_negative_rolling'):
    signal_col = 'sentiment_volume'

    # Initialize the signal column
    df[signal_col] = 0

    # Generate buy signals based on positive rolling counts
    buy_condition = (
        (df[slope_positive_col] > 0) & 
        (df[sma_positive_col] > 0) & 
        (df[price_rsi_col] < price_rsi_low)
    )
    df.loc[buy_condition, signal_col] = 'buy'

    # Generate sell signals based on negative rolling counts
    sell_condition = (
        (df[slope_negative_col] < 0) & 
        (df[sma_negative_col] < 0) & 
        (df[price_rsi_col] > price_rsi_high)
    )
    df.loc[sell_condition, signal_col] = 'sell'

    # Replace '' and NaN values with 0
    df[signal_col] = df[signal_col].replace({'': 0}).fillna(0)

    # Convert 'buy' and 'sell' to 1 and -1
    df[signal_col] = df[signal_col].replace({'buy': 1, 'sell': -1})

    return df


def generate_combined_signals(df):
    # List of signal prefixes to combine
    signal_prefixes = ['sma', 'rsi', 'slope', 'roc']
    all_signal_columns = []
    
    for prefix in signal_prefixes:
        # Find all signal columns that contain the prefix
        signal_columns = [col for col in df.columns if f'signal_{prefix}' in col]
        
        # Ensure '' and NaN values are converted to 0, and 'buy' and 'sell' are converted to 1 and -1
        for col in signal_columns:
            df[col] = df[col].replace({'': 0, 'buy': 1, 'sell': -1}).fillna(0)
        
        # Calculate the combined signal
        combined_signal = df[signal_columns].sum(axis=1)
        
        # Create the combined signal column
        combined_signal_name = f'{prefix}_signal_combine'
        df[combined_signal_name] = combined_signal.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

        # Collect all signal columns for the final combined signal
        all_signal_columns.extend(signal_columns)

    # Calculate the combined signal for all signals
    combined_all_signal = df[all_signal_columns].sum(axis=1)
    df['combined_all_signal'] = combined_all_signal.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        
    return df


# --------------------------------------------------
# ----------- Trading indicators lite functions => Create more signals with no so restrictive condicitons
# --------------------------------------------------
def generate_rsi_signals_lite(df, sentiment_col, positive_threshold=0.1, negative_threshold=-0.1, price_rsi_low=40, price_rsi_high=60):
    signal_col = f'signal_rsi_lite_{sentiment_col}'

    # Initialize the signal column
    df[signal_col] = 0

    df.loc[(df['rsi'] < price_rsi_low) & (df[sentiment_col] > positive_threshold), signal_col] = 'buy'
    df.loc[(df['rsi'] > price_rsi_high) & (df[sentiment_col] < negative_threshold), signal_col] = 'sell'

    # Replace '' and NaN values with 0
    df[signal_col] = df[signal_col].replace({'': 0}).fillna(0)

    # Convert 'buy' and 'sell' to 1 and -1
    df[signal_col] = df[signal_col].replace({'buy': 1, 'sell': -1})
    
    return df


def generate_sma_signals_lite(df, sentiment_col, fast_window=5, slow_window=20, positive_threshold=0.1, negative_threshold=-0.1):
    # Calculate fast and slow SMAs for the price
    df['fast_sma'] = df['Close'].rolling(window=fast_window).mean()
    df['slow_sma'] = df['Close'].rolling(window=slow_window).mean()
    
    # Generate buy and sell signals
    signal_col = f'signal_sma_lite_{sentiment_col}'
    df[signal_col] = 0
    
    df.loc[(df['fast_sma'] > df['slow_sma']) & 
           (df['fast_sma'].shift(1) <= df['slow_sma'].shift(1)) & 
           (df[sentiment_col] > positive_threshold), signal_col] = 'buy'
    
    df.loc[(df['fast_sma'] < df['slow_sma']) & 
           (df['fast_sma'].shift(1) >= df['slow_sma'].shift(1)) & 
           (df[sentiment_col] < negative_threshold), signal_col] = 'sell'
    
    # Replace '' and NaN values with 0
    df[signal_col] = df[signal_col].replace({'': 0}).fillna(0)

    # Convert 'buy' and 'sell' to 1 and -1
    df[signal_col] = df[signal_col].replace({'buy': 1, 'sell': -1})
    
    return df


def generate_roc_signals_lite(df, sentiment_col, positive_threshold=0.1, negative_threshold=-0.1):
    signal_col = f'signal_roc_lite_{sentiment_col}'

    # Initialize the signal column
    df[signal_col] = 0

    # Generate buy signal: Detect turning points in ROC
    buy_condition = (
        (df['roc'] > 0) &
        (df['roc'].shift(2) <= 0) &
        (df[sentiment_col] > positive_threshold)
    )
    df.loc[buy_condition, signal_col] = 'buy'
    
    # Generate sell signal: Detect turning points in ROC
    sell_condition = (
        (df['roc'] < 0) &
        (df['roc'].shift(2) >= 0) &
        (df[sentiment_col] < negative_threshold)
    )
    df.loc[sell_condition, signal_col] = 'sell'

    # Replace '' and NaN values with 0
    df[signal_col] = df[signal_col].replace({'': 0}).fillna(0)

    # Convert 'buy' and 'sell' to 1 and -1
    df[signal_col] = df[signal_col].replace({'buy': 1, 'sell': -1})
    
    return df


def generate_slope_signals_lite(df, sentiment_col, positive_threshold=0.1, negative_threshold=-0.1):
    signal_col = f'signal_slope_lite_{sentiment_col}'

    # Initialize the signal column
    df[signal_col] = 0

    # Generate buy signal: Detect turning points in slope
    buy_condition = (
        (df['slope'] > 0) &
        (df['slope'].shift(2) <= 0) &
        (df[sentiment_col] > positive_threshold)
    )
    df.loc[buy_condition, signal_col] = 'buy'
    
    # Generate sell signal: Detect turning points in slope
    sell_condition = (
        (df['slope'] < 0) &
        (df['slope'].shift(2) >= 0) &
        (df[sentiment_col] < negative_threshold)
    )
    df.loc[sell_condition, signal_col] = 'sell'

    # Replace '' and NaN values with 0
    df[signal_col] = df[signal_col].replace({'': 0}).fillna(0)

    # Convert 'buy' and 'sell' to 1 and -1
    df[signal_col] = df[signal_col].replace({'buy': 1, 'sell': -1})
    
    return df

# ---------------------- Signals Analysis ------------------------
def analyze_signals(df, percentage_threshold=0.15):
    # Consider all signal columns
    signal_columns = [col for col in df.columns if 'signal' in col]
    total_data_points = len(df)
    min_signals = int(total_data_points * percentage_threshold)
    analysis_results = []

    for signal_col in signal_columns:
        buy_signals = (df[signal_col] == 1).sum()
        sell_signals = (df[signal_col] == -1).sum()
        total_signals = buy_signals + sell_signals

        analysis_results.append({
            'Signal': signal_col,
            'Buy Signals': buy_signals,
            'Sell Signals': sell_signals,
            'Total Signals': total_signals
        })

    analysis_df = pd.DataFrame(analysis_results)
    
    # Check if the number of signals is sufficient
    analysis_df['Sufficient Signals'] = analysis_df['Total Signals'] >= min_signals

    # Provide a summary message
    sufficient_signals_count = analysis_df['Sufficient Signals'].sum()
    total_signal_types = len(analysis_df)

    if sufficient_signals_count >= total_signal_types:
        summary_message = f'All signal types have sufficient signals (Threshold: {min_signals} signals).'
    else:
        summary_message = f'{sufficient_signals_count} out of {total_signal_types} signal types have sufficient signals (Threshold: {min_signals} signals).'

    return analysis_df, summary_message


# --------------------------------------------------
#  Feature engineering functions 
# --------------------------------------------------

def generate_lagged_features(df_stock, scoring_columns, price_indicators_columns, sentiment_indicators_columns, shifts=[1, 3, 5]):

    # Initialize a list to hold new feature columns
    shifted_feature_columns = []

    # Apply shifts to scoring columns
    for col in scoring_columns:
        for shift in shifts:
            shifted_col_name = f'{col}_shift_{shift}'
            df_stock[shifted_col_name] = df_stock[col].shift(shift)
            shifted_feature_columns.append(shifted_col_name)

    # Apply shifts to price indicators columns
    for col in price_indicators_columns:
        for shift in shifts:
            shifted_col_name = f'{col}_shift_{shift}'
            df_stock[shifted_col_name] = df_stock[col].shift(shift)
            shifted_feature_columns.append(shifted_col_name)

    # Apply shifts to sentiment indicators columns
    for col in sentiment_indicators_columns:
        for shift in shifts:
            shifted_col_name = f'{col}_shift_{shift}'
            df_stock[shifted_col_name] = df_stock[col].shift(shift)
            shifted_feature_columns.append(shifted_col_name)

    return df_stock, shifted_feature_columns


# --------------------------------------------------
#  Backtesting functions 
# --------------------------------------------------

# Rolling window 
def generate_rolling_windows(data_length, min_train_window=50, split_ratio=0.8, max_windows=10):
    test_window = ceil(min_train_window * (1 - split_ratio))
    max_train_window = (data_length - (max_windows * test_window)) // max_windows
    train_window = max(min_train_window, max_train_window)
    num_splits = min(max_windows, (data_length - train_window) // test_window)
    
    windows_df = pd.DataFrame(columns=['Iteration', 'Train Start', 'Train End', 'Test Start', 'Test End'])
    plot_idx = 0
    start_index = 0

    while start_index + train_window + test_window <= data_length and plot_idx < num_splits:
        plot_idx += 1
        train_start = start_index
        train_end = train_start + train_window - 1
        test_start = train_end + 1
        test_end = min(test_start + test_window - 1, data_length - 1)
        
        windows_df = pd.concat([windows_df, pd.DataFrame({
            'Iteration': [plot_idx],
            'Train Start': [train_start],
            'Train End': [train_end],
            'Test Start': [test_start],
            'Test End': [test_end]
        })], ignore_index=True)
        
        start_index = test_start

    if windows_df.iloc[-1]['Test End'] < data_length - 1:
        windows_df.iloc[-1, windows_df.columns.get_loc('Test End')] = data_length - 1
    
    return windows_df


# Function to calculate final return based on model predictions
def calculate_final_return(df, pred_col, actual_col, condition_col, threshold=None):
    if threshold is not None:
        condition = df[condition_col] > threshold
    else:
        condition = df[condition_col] > 0

    df['Strategy Returns'] = df[actual_col].pct_change().shift(-1) * condition
    df['Strategy Returns'] = df['Strategy Returns'].dropna()  # Drop initial NaN values
    
    # Calculate cumulative returns and get the final return
    final_return = (1 + df['Strategy Returns']).prod() - 1 if not df['Strategy Returns'].empty else None
    return final_return


# Function to calculate Buy and Hold final return
def calculate_final_buy_and_hold_final_return(df, close_col):
    if df.empty or close_col not in df.columns:
        raise ValueError("DataFrame is empty or column not found")

    df = df.dropna(subset=[close_col])
    df['Buy and Hold Returns'] = df[close_col].pct_change()
    df['Buy and Hold Returns'] = df['Buy and Hold Returns'].dropna()  # Drop initial NaN values
    
    # Calculate cumulative returns and get the final return
    final_buy_and_hold_return = (1 + df['Buy and Hold Returns']).prod() - 1 if not df['Buy and Hold Returns'].empty else None
    return final_buy_and_hold_return


# Function to calculate rolling quantile threshold
def calculate_rolling_quantile(df, pred_col, quantile=0.95, window=252):
    if pred_col not in df.columns:
        raise ValueError(f"Column {pred_col} not found in DataFrame")
    rolling_quantile = df[pred_col].rolling(window=window, min_periods=1).quantile(quantile)
    return rolling_quantile


# --------------------------------------------------
# White Reality Check functions
# --------------------------------------------------
def detrend_series(series):
    return series - np.mean(series)


def bootstrap_series(series, n_iterations=1000):
    bootstrapped_series = []
    for _ in range(n_iterations):
        sample = np.random.choice(series, size=len(series), replace=True)
        bootstrapped_series.append(sample)
    return np.array(bootstrapped_series)


def calculate_performance(series):
    return np.prod(1 + series) - 1


def white_reality_check(returns, n_iterations=1000):
    detrended_returns = detrend_series(returns)
    bootstrapped_returns = bootstrap_series(detrended_returns, n_iterations)
    bootstrapped_performances = np.array([calculate_performance(bootstrapped) for bootstrapped in bootstrapped_returns])

    original_performance = calculate_performance(returns)
    p_value = np.mean(bootstrapped_performances >= original_performance)

    return original_performance, p_value


# #########################
# Main function
# #########################

def main():

    # #########################
    # Get the stocks data
    # #########################
    
    # ----------- Get stocks list -----------
    print("Getting stocks list...")
    stocks = get_russell3000_tickers()

    # stocks = stocks[1:2]

    # ----------- Get stocks price data  -----------
    print("Getting stocks data...")
    df_stocks_all = get_stocks_data_hist_1h(stocks)
    # Save
    df_stocks_all.to_csv('data/stocks_data.csv', index=False)

    # ----------- Get stocks news data  -----------
    print("Getting news data...")

    # Setup client
    api_key = 'cpg51cpr01ql1vn3hs30cpg51cpr01ql1vn3hs3g'
    finnhub_client = finnhub.Client(api_key=api_key)
    
    # Get the news
    get_all_news(stocks,finnhub_client)


    # #####################################
    # Generate Sentiment Indicators
    # #####################################
    print("Generate Sentiment Indicators...")


    for SYMBOL in stocks:

        try: 

            print(f"-"*50)
            print(f"Generating Sentiment Indicators for symbol {SYMBOL}")
            print(f"-"*50)
            
            # -----------------------
            # Load symbol data 
            # -----------------------
            # Load the data
            df_stock = df_stocks_all[df_stocks_all['Ticker'] == SYMBOL].copy()

            df_stock.reset_index(inplace=True)
            df_stock.rename(columns={'index': 'Datetime'}, inplace=True)

            # Load the news
            df_news = pd.read_csv(f'data/news_{SYMBOL}.csv')

            
            # -----------------------
            # Sentiment Analysis 
            # -----------------------

            # Perform sentiment analysis
            df_sentiment = analyze_sentiment(df_news)

            # sort df_sentiment by datetime asc
            df_sentiment = df_sentiment.sort_values('datetime')

            # Apply aggregation functions
            df_avg_sentiment = aggregate_average_sentiment(df_sentiment, freq='h')
            df_weighted_sentiment = aggregate_sentiment_weighted(df_sentiment, freq='h')
            df_sum_sentiment = aggregate_sum_sentiment(df_sentiment, freq='h')
            df_extreme_sentiment = sentiment_extreme(df_sentiment, freq='h')

            # Apply classification and aggregation function
            df_classified_sentiment = classify_and_aggregate_sentiment(df_sentiment, freq='h', positive_threshold=0.1, negative_threshold=-0.1, window_size=4)

            # Combine the results into a single DataFrame for further analysis
            df_all_aggregated = df_avg_sentiment.copy()
            df_all_aggregated = df_all_aggregated.merge(df_weighted_sentiment, on='datetime', how='left')
            df_all_aggregated = df_all_aggregated.merge(df_sum_sentiment, on='datetime', how='left')
            df_all_aggregated = df_all_aggregated.merge(df_extreme_sentiment, on='datetime', how='left')
            df_all_aggregated = df_all_aggregated.merge(df_classified_sentiment, on='datetime', how='left')

            
            # -----------------------------
            # Sentiment score indicators 
            # -----------------------------

            # Columns to apply indicators
            columns_to_apply = [
                'avg_headline_sentiment', 'avg_summary_sentiment',
                'weighted_headline_sentiment', 'weighted_summary_sentiment',
                'sum_headline_sentiment', 'sum_summary_sentiment',
                'extreme_headline_sentiment', 'extreme_summary_sentiment',
                'positive_rolling', 'negative_rolling', 'neutral_rolling'
            ]

            # Apply indicators to all relevant columns
            for column in columns_to_apply:
                df_all_aggregated = calculate_roc(df_all_aggregated, column=column, window=12)
                df_all_aggregated = calculate_rsi(df_all_aggregated, column=column, window=14)
                df_all_aggregated = calculate_moving_average(df_all_aggregated, column=column, window=5)
                df_all_aggregated = calculate_slope(df_all_aggregated, column=f'{column}', window=5)

            # keep only dateeim from 2023 
            df_all_aggregated = df_all_aggregated[df_all_aggregated['datetime'] > '2023-01-01']
            

            # -----------------------------
            # Correlation Indicators 
            # -----------------------------

            # Prepare the data for correlation analysis
            # Ensure both datetime columns are in the same timezone
            df_stock['Datetime'] = df_stock['Datetime'].dt.tz_convert('UTC')
            df_all_aggregated['datetime'] = df_all_aggregated['datetime'].dt.tz_localize('UTC')

            # Truncate or reset the minutes and seconds to 00:00 to match hourly level
            df_stock['Datetime'] = df_stock['Datetime'].dt.floor('H')
            df_all_aggregated['datetime'] = df_all_aggregated['datetime'].dt.floor('H')

            # Merge the data
            df_stock = df_stock.merge(df_all_aggregated, how='inner', left_on='Datetime', right_on='datetime')

            # Calculate the percentage change
            df_stock['pct_change'] = df_stock['Close'].pct_change()

            # Calculate the rolling correlation
            df_stock['rolling_correlation'], df_stock['rolling_p_value'] = rolling_pearsonr(df_stock['avg_headline_sentiment'], df_stock['pct_change'], 5)

            # Define the trading signal
            df_stock['significant_correlation'] = (df_stock['rolling_p_value'] < 0.05) & (abs(df_stock['rolling_correlation']) > 0.5)
            

            # -----------------------------
            #  Price Indicators 
            # -----------------------------

            df_stock['rsi'] = ta.rsi(df_stock['Close'], length=14)
            df_stock['sma'] = ta.sma(df_stock['Close'], length=12)
            df_stock['roc'] = ta.roc(df_stock['Close'], length=12)
            df_stock['slope'] = ta.slope(df_stock['sma'], length=5)

            
            # ------------ Prepare output data ------------

            # Combine all stocks data
            if 'df_stocks_with_indicators' not in globals():
                df_stocks_with_indicators = df_stock
            else:
                df_stocks_with_indicators = pd.concat([df_stocks_with_indicators, df_stock])

        except:
            print(f"Error in symbol {SYMBOL}")
            # remove the symbol from the list
            stocks.remove(SYMBOL)

        # Save the data
        df_stocks_with_indicators.to_csv('data/stocks_with_indicators.csv', index=False)


    # ##################
    # Trading Signals
    # ##################

    print("Generate Trading Signals...")
   
    stocks = df_stocks_with_indicators['Ticker'].unique()

    for SYMBOL in stocks:

        try: 
                
            print(f"-"*50)
            print(f"Generating Trading Signals for symbol {SYMBOL}")
            print(f"-"*50)

            SYMBOL = stocks[0]

            # ------------ Load the data ---------------
            df_stock = df_stocks_with_indicators[df_stocks_with_indicators['Ticker'] == SYMBOL].copy()


            # -----------------------------
            # Geerate Trading Signals 
            # -----------------------------

            # Define news column to generate signals 
            sentiment_columns = [
                'avg_headline_sentiment', 'avg_summary_sentiment',
                'weighted_headline_sentiment', 'weighted_summary_sentiment',
                'sum_headline_sentiment', 'sum_summary_sentiment',
                'extreme_headline_sentiment', 'extreme_summary_sentiment'
            ]
            
            # Define the thresholds for positive, negative, and neutral sentiment
            threshold_positive = 0.1
            threshold_negative = 0.1

            # Generate trading signals
            for sentiment_col in sentiment_columns:
                df_stock = generate_rsi_signals(df_stock, sentiment_col=sentiment_col, positive_threshold=threshold_positive, negative_threshold=-threshold_negative, price_rsi_low=40, price_rsi_high=55, sentiment_rsi_low=45, sentiment_rsi_high=55)
                df_stock = generate_sma_signals(df_stock, sentiment_col=sentiment_col, positive_threshold=threshold_positive, negative_threshold=-threshold_negative, fast_window=5, slow_window=20)
                df_stock = generate_roc_signals(df_stock, sentiment_col=sentiment_col, positive_threshold=threshold_positive, negative_threshold=-threshold_negative)
                df_stock = generate_slope_signals(df_stock, sentiment_col=sentiment_col, positive_threshold=threshold_positive, negative_threshold=-threshold_negative)
                df_stock = generate_rsi_signals_lite(df_stock, sentiment_col=sentiment_col, positive_threshold=threshold_positive, negative_threshold=-threshold_negative, price_rsi_low=45, price_rsi_high=55)
                df_stock = generate_sma_signals_lite(df_stock, sentiment_col=sentiment_col, positive_threshold=threshold_positive, negative_threshold=-threshold_negative, fast_window=5, slow_window=20)
                df_stock = generate_roc_signals_lite(df_stock, sentiment_col=sentiment_col, positive_threshold=threshold_positive, negative_threshold=-threshold_negative)
                df_stock = generate_slope_signals_lite(df_stock, sentiment_col=sentiment_col, positive_threshold=threshold_positive, negative_threshold=-threshold_negative)


            # Generate sentiment volume signals
            df_stock = generate_sentiment_volume_signals(
                        df_stock,
                        price_rsi_col='rsi', 
                        price_rsi_low=45, 
                        price_rsi_high=55, 
                        slope_positive_col='slope_positive_rolling', 
                        sma_positive_col='sma_positive_rolling', 
                        slope_negative_col='slope_negative_rolling', 
                        sma_negative_col='sma_negative_rolling'
                        )
            

            # Generate combined signals
            df_stock = generate_combined_signals(df_stock)

            # Analyze the signals
            analysis_df, summary_message = analyze_signals(df_stock, percentage_threshold=0.15)


            # ----------- Save the analysis results ------------
            # Combine all stocks data with signals
            if 'df_stocks_with_trading_signals' not in locals():
                df_stocks_with_trading_signals = df_stock
            else:
                df_stocks_with_trading_signals = pd.concat([df_stocks_with_trading_signals, df_stock])

        except:
            print(f"Error in symbol {SYMBOL}")
            # remove the symbol from the list
            stocks.remove(SYMBOL)


    # Save trading signals
    print("Saving results...")

    # Save the data
    df_stocks_with_trading_signals.to_csv('data/stocks_with_trading_signals.csv', index=False)


    # ########################################################
    # Gererate Regression Predictive Machine Learning Model
    # ########################################################

    stocks = df_stocks_with_trading_signals['Ticker'].unique()

    # Initialize lists to store all results
    all_predictions = []
    all_results = []

    # Iterate through each symbol
    for SYMBOL in stocks:
        print(f"-"*50)
        print(f"Performing Walk-Forward Optimization for symbol {SYMBOL}")
        print(f"-"*50)

        # ------- Load the data ---------
        df_stock = df_stocks_with_trading_signals[df_stocks_with_trading_signals['Ticker'] == SYMBOL].copy()

        # Replace '' with 0 and fill NaN values with 0 and infinity with 0
        df_stock = df_stock.replace('', 0)
        df_stock = df_stock.fillna(0)
        df_stock = df_stock.replace([np.inf, -np.inf], 0)

        
        # ----------------------------
        # Feature Engineering  
        # ----------------------------

        # Define the target
        target_distance_ahead = 20
        df_stock['target'] = df_stock['Close'].pct_change().shift(-target_distance_ahead)
        df_stock = df_stock.dropna(subset=['target'])

        # Define sentinemt scoring features
        scoring_columns = [
            'avg_headline_sentiment', 'avg_summary_sentiment',
            'weighted_headline_sentiment', 'weighted_summary_sentiment',
            'sum_headline_sentiment', 'sum_summary_sentiment',
            'extreme_headline_sentiment', 'extreme_summary_sentiment'
        ]
        
        # Define price indicators features
        price_indicators_columns = ['rsi', 'sma', 'roc', 'slope']

        # Define sentiment indicators features
        sentiment_indicators_columns = [
            'roc_extreme_summary_sentiment', 'rsi_extreme_summary_sentiment', 'sma_extreme_summary_sentiment',
            'slope_extreme_summary_sentiment', 'roc_positive_rolling', 'rsi_positive_rolling', 'sma_positive_rolling',
            'slope_positive_rolling', 'roc_negative_rolling', 'rsi_negative_rolling', 'sma_negative_rolling',
            'slope_negative_rolling'
        ]

        # Define lagged features
        shifts = [1, 3, 5]
        df_stock, shifted_feature_columns = generate_lagged_features(df_stock, scoring_columns, price_indicators_columns, sentiment_indicators_columns, shifts=shifts)

        # Add trading signals as features
        feature_columns =   (scoring_columns + 
                            price_indicators_columns + 
                            sentiment_indicators_columns + 
                            shifted_feature_columns +
                            [col for col in df_stock.columns if 'signal' in col])
        

        # Drop rows with NaN values created by shifting
        df_stock = df_stock.dropna(subset=feature_columns + ['target'])
        
        # Prepare feature datasets for training
        X = df_stock[feature_columns]
        y = df_stock['target']
        y = y[X.index]
        

        # ----------------------------
        # Model Tradining 
        # ----------------------------

        # Walk Forward Optimization split
        windows_df = generate_rolling_windows(len(X), min_train_window=50, split_ratio=0.8, max_windows=10)

        # Loop through each window to train and evaluate the model
        symbol_predictions = []
        symbol_results = []

        for _, row in windows_df.iterrows():
                       
            train_start = int(row['Train Start'])
            train_end = int(row['Train End'])
            test_start = int(row['Test Start'])
            test_end = int(row['Test End'])

            X_train, X_test = X.iloc[train_start:train_end+1], X.iloc[test_start:test_end+1]
            y_train, y_test = y.iloc[train_start:train_end+1], y.iloc[test_start:test_end+1]

            # Replace infinity with NaN
            X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
            X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Drop rows with NaN values
            X_train.dropna(inplace=True)
            X_test.dropna(inplace=True)
            y_train = y_train[X_train.index]
            y_test = y_test[X_test.index]

            # ------- Feature Selection ---------
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)

            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1]

            N = 20
            top_indices = indices[:N]
            top_features = [X.columns[i] for i in top_indices]

            X_train_reduced = X_train[top_features]
            X_test_reduced = X_test[top_features]

            # ------- Train Models ---------
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train_reduced, y_train)
            y_test_pred_rf = rf.predict(X_test_reduced)

            model_xgb = XGBRegressor(n_estimators=100, random_state=42)
            model_xgb.fit(X_train_reduced, y_train)
            y_test_pred_xgb = model_xgb.predict(X_test_reduced)

            model_lr = LinearRegression()
            model_lr.fit(X_train_reduced, y_train)
            y_test_pred_lr = model_lr.predict(X_test_reduced)

            y_test_pred_ensemble = (y_test_pred_rf + y_test_pred_xgb + y_test_pred_lr) / 3


            # ------------ Save Predictions ---------------
            # Save predictions for each model and the ensemble
            symbol_predictions.append(pd.DataFrame({
                'Symbol': SYMBOL,
                'Datetime': df_stock.iloc[test_start:test_end+1].index,
                'RF Prediction': y_test_pred_rf,
                'XGB Prediction': y_test_pred_xgb,
                'LR Prediction': y_test_pred_lr,
                'Ensemble Prediction': y_test_pred_ensemble
            }))

            r2_train_rf = r2_score(y_train, rf.predict(X_train_reduced))
            r2_test_rf = r2_score(y_test, y_test_pred_rf)
            r2_train_xgb = r2_score(y_train, model_xgb.predict(X_train_reduced))
            r2_test_xgb = r2_score(y_test, y_test_pred_xgb)
            r2_train_lr = r2_score(y_train, model_lr.predict(X_train_reduced))
            r2_test_lr = r2_score(y_test, y_test_pred_lr)
            r2_train_ensemble = r2_score(y_train, (rf.predict(X_train_reduced) + model_xgb.predict(X_train_reduced) + model_lr.predict(X_train_reduced)) / 3)
            r2_test_ensemble = r2_score(y_test, y_test_pred_ensemble)

            symbol_results.append({
                'Iteration': row['Iteration'],
                'Symbol': SYMBOL,
                'Train R2 RF': r2_train_rf,
                'Test R2 RF': r2_test_rf,
                'Train R2 XGB': r2_train_xgb,
                'Test R2 XGB': r2_test_xgb,
                'Train R2 LR': r2_train_lr,
                'Test R2 LR': r2_test_lr,
                'Train R2 Ensemble': r2_train_ensemble,
                'Test R2 Ensemble': r2_test_ensemble
            })

            # Combine results for this symbol
            symbol_predictions_df = pd.concat(symbol_predictions).reset_index(drop=True)
            symbol_results_df = pd.DataFrame(symbol_results)

            # Save the results
            symbol_predictions_df.to_csv(f'data/{SYMBOL}_predictions.csv', index=False)
            symbol_results_df.to_csv(f'data/{SYMBOL}_results.csv', index=False)

        # Merge predictions with the stock data for this symbol
        df_stock = pd.merge(df_stock, symbol_predictions_df, how='left', left_index=True, right_on='Datetime')

        # Save the data
        df_stock.to_csv(f'data/{SYMBOL}_ml_returns.csv', index=False)

        # Append to combined results
        all_predictions.append(df_stock)
        all_results.append(symbol_results_df)

        # Combine all predictions and results for all symbols
        combined_predictions_df = pd.concat(all_predictions).reset_index(drop=True)
        combined_results_df = pd.concat(all_results).reset_index(drop=True)


    # Save the results
    combined_predictions_df.to_csv('data/combined_predictions.csv', index=False)
    combined_results_df.to_csv('data/combined_results.csv', index=False)

   
    # #########################
    # Backtest Analysis
    # #########################

    print("Generate Returns...")
    import glob
    # load the data
    # Define the pattern to search for CSV files
    pattern = 'data/*_ml_returns*.csv'

    # Use glob to find all files that match the pattern
    csv_files = glob.glob(pattern)

    # Read each CSV file into a DataFrame and store them in a list
    df_list = [pd.read_csv(file) for file in csv_files]

    # Concatenate all DataFrames in the list into a single DataFrame
    combined_predictions_df = pd.concat(df_list, ignore_index=True)
    stocks = combined_predictions_df['Ticker'].unique()
    
    # List of models to analyze
    model_columns = ['RF Prediction', 'XGB Prediction', 'LR Prediction', 'Ensemble Prediction']

    stocks = combined_predictions_df['Ticker'].unique()

    # Placeholder for storing results
    returns_analysis = []
    best_stocks = []

    for SYMBOL in stocks:
        print(f"{'-'*50}")
        print(f"Analyzing Returns for {SYMBOL}")
        print(f"{'-'*50}")

        # Load data
        df_stock = combined_predictions_df[combined_predictions_df['Ticker'] == SYMBOL].copy()

        # Check if df_stock is empty
        if df_stock.empty:
            print(f"No data found for {SYMBOL}")
            continue

        # Calculate Buy and Hold final return
        try:
            final_buy_and_hold_return = calculate_final_buy_and_hold_final_return(df_stock, 'Close')
        except ValueError as e:
            print(e)
            continue

        # Iterate through each model's predictions
        for model in model_columns:
            if model not in df_stock.columns:
                print(f"Model column {model} not found in data for {SYMBOL}")
                continue

            # Calculate final return for basic strategy (predicted return > 0)
            final_basic_return = calculate_final_return(df_stock, model, 'Close', model)
            
            # Calculate rolling quantile threshold
            try:
                df_stock[f'{model}_quantile_threshold'] = calculate_rolling_quantile(df_stock, model)
            except ValueError as e:
                print(e)
                continue

            # Calculate final return for strategy with quantile threshold
            final_quantile_return = calculate_final_return(df_stock, model, 'Close', model, df_stock[f'{model}_quantile_threshold'])

            # Trend-following strategy
            df_stock['Trend Signal'] = (df_stock[model] > 0) & (df_stock['slope'] > 0)
            df_stock['Trend Strategy Returns'] = df_stock['Close'].pct_change().shift(-1) * df_stock['Trend Signal']
            df_stock['Trend Strategy Returns'] = df_stock['Trend Strategy Returns'].dropna()  # Drop initial NaN values
            final_trend_return = (1 + df_stock['Trend Strategy Returns']).prod() - 1 if not df_stock['Trend Strategy Returns'].empty else None

            # Mean-reverting strategy
            df_stock['Mean Reversion Signal'] = (df_stock[model] > 0) & (df_stock['rsi'] < 30)
            df_stock['Mean Reversion Strategy Returns'] = df_stock['Close'].pct_change().shift(-1) * df_stock['Mean Reversion Signal']
            df_stock['Mean Reversion Strategy Returns'] = df_stock['Mean Reversion Strategy Returns'].dropna()  # Drop initial NaN values
            final_mean_reversion_return = (1 + df_stock['Mean Reversion Strategy Returns']).prod() - 1 if not df_stock['Mean Reversion Strategy Returns'].empty else None

            # Store the results
            returns_analysis.append({
                'Symbol': SYMBOL,
                'Model': model,
                'Final Buy and Hold Return': final_buy_and_hold_return,
                'Final Basic Strategy Return': final_basic_return,
                'Final Quantile Strategy Return': final_quantile_return,
                'Final Trend Strategy Return': final_trend_return,
                'Final Mean Reversion Strategy Return': final_mean_reversion_return
            })

        returns_df = pd.DataFrame(returns_analysis)

        # Print the stocks with returns better than buy and hold
        better_than_buy_and_hold = returns_df[(returns_df['Final Basic Strategy Return'] > returns_df['Final Buy and Hold Return']) |
                                        (returns_df['Final Quantile Strategy Return'] > returns_df['Final Buy and Hold Return']) |
                                        (returns_df['Final Trend Strategy Return'] > returns_df['Final Buy and Hold Return']) |
                                        (returns_df['Final Mean Reversion Strategy Return'] > returns_df['Final Buy and Hold Return'])]
       
    
    best_stocks = better_than_buy_and_hold['Symbol'].unique()


    # #########################
    # White Reality Check
    # #########################

    # Initialize list to store results
    returns_analysis = []

    # Iterate through each stock and model
    for SYMBOL in best_stocks:
        df_stock = combined_predictions_df[combined_predictions_df['Ticker'] == SYMBOL].copy()
        
        for model in model_columns:
            df_stock['Strategy Returns'] = df_stock['Close'].pct_change().shift(-1) * (df_stock[model] > 0)
            returns = df_stock['Strategy Returns'].dropna()
            
            performance, p_value = white_reality_check(returns)
            
            returns_analysis.append({
                'Symbol': SYMBOL,
                'Model': model,
                'Performance': performance,
                'p-value': p_value,
                'Buy and Hold Return': final_buy_and_hold_return
            })

    # Convert results to a DataFrame and filter
    returns_df = pd.DataFrame(returns_analysis)
    significant_models = returns_df[returns_df['p-value'] < 0.05]
    print(significant_models)


    # #########################
    # Plot Significant models
    # #########################

    # Plot cumulative returns for significant models
    for _, row in significant_models.iterrows():
        symbol = row['Symbol']
        model = row['Model']
        
        df_stock = combined_predictions_df[combined_predictions_df['Ticker'] == symbol].copy()
        df_stock['Strategy Returns'] = df_stock['Close'].pct_change().shift(-1) * (df_stock[model] > 0)
        
        # Calculate cumulative returns
        df_stock['Cumulative Returns'] = (1 + df_stock['Strategy Returns']).cumprod() - 1
        
        # Prepare data for plotting
        df_cumulative_returns = df_stock[['Datetime_x', 'Cumulative Returns']].dropna()
        
        # Create Plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_cumulative_returns['Datetime_x'], y=df_cumulative_returns['Cumulative Returns'],
                                mode='lines', name=f'{symbol} - {model}'))
        
        fig.update_layout(title=f'Cumulative Returns for {symbol} using {model}',
                        xaxis_title='Date', yaxis_title='Cumulative Returns',
                        template='plotly_dark')
        
        # Save or show plot
        plot_dir = 'plots'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        fig.write_image(f'{plot_dir}/cumulative_returns_{symbol}_{model}.png')
        fig.write_html(f'{plot_dir}/cumulative_returns_{symbol}_{model}.html')
        fig.show()



    # ################# 
    # Finish
    # #################

    print("Done!")


# #########################
# Run
# #########################

if __name__ == '__main__':
    main()