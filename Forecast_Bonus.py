
# -------------------------
# Import libraries 
# -------------------------

import pandas as pd 
import yfinance as yf
import numpy as np

import os
import pandas as pd

import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = "browser"

from darts import TimeSeries
from darts.models import NBEATSModel


# -------------------------
# Define functions
# -------------------------

# ----------- Tickers collection functions -----------

def get_spx500_stocks():
    """
    Get the list of stocks from wikipedia
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    stocks = pd.read_html(url, header=0)[0]
    return stocks['Symbol'].tolist()


# ----------- Price data collection functions -----------

def get_stocks_monthly_data(tickers):
    all_data = []
    
    for ticker in tickers:
        data = yf.download(ticker, period="5y", interval="1mo")
        data['Ticker'] = ticker
        all_data.append(data)
    
    # Concatenate all dataframes
    combined_data = pd.concat(all_data)
    
    # Reset index to have 'Date' as a column
    combined_data.reset_index(inplace=True)
    
    return combined_data


# -------------------------
# Main function
# -------------------------

def main():

    # -------------------------
    # Get the stocks data
    # -------------------------
    
    # ----------- Get stocks list -----------
    print("Getting stocks list...")
    stocks = get_spx500_stocks()

    # stocks = stocks[1:2]

    # ----------- Get stocks price data  -----------
    print("Getting stocks data...")
    df_stocks_all = get_stocks_monthly_data(stocks)
    # Save
    df_stocks_all.to_csv('data/stocks_data_monthly.csv', index=False)


    # --------------------------------------------------------
    # Gererate Regression Predictive Machine Learning Model
    # --------------------------------------------------------

    # Iterate through each symbol
    for SYMBOL in stocks:
        print(f"-"*50)
        print(f"Performing forecast for {SYMBOL}")
        print(f"-"*50)

        # ------- Load the data ---------
        df_stock = df_stocks_all[df_stocks_all['Ticker'] == SYMBOL].copy()

        # Replace '' with 0 and fill NaN values with 0 and infinity with 0
        df_stock = df_stock.replace('', 0)
        df_stock = df_stock.fillna(0)
        df_stock = df_stock.replace([np.inf, -np.inf], 0)


        # ------- Forecast 6 months ahead using NBeats ---------
        # Define the forecast horizon
        forecast_horizon = 6

        df_dart = df_stock[['Date', 'Close']]

        df_dart['Date'] = pd.to_datetime(df_dart['Date'])

        # Create a Darts TimeSeries object from the 'Consumption' column
        series = TimeSeries.from_dataframe(df_dart.set_index('Date'), value_cols=['Close'])


        # ----- Train model ------
        model_nbeats = NBEATSModel(input_chunk_length=forecast_horizon, output_chunk_length=20)

        # Fit the model to the series
        model_nbeats.fit(series)


        # ----- Make predictions ------
        # Make a prediction for the next 365 days
        prediction = model_nbeats.predict(n=forecast_horizon)

        # transdorm prediction darts object into a pd.DataFrame
        df_pred = pd.DataFrame({'ds': prediction.time_index, 'y': prediction.univariate_values(0)})


        # ------- Combine the actual and forecast data ------
        # Rename the columns
        df_predictions = df_pred.copy()
        df_predictions.rename(columns={'ds': 'Date', 'y': 'Close'}, inplace=True)

        # Add Model column
        df_predictions['Type'] = 'Forecast'
        df_dart['Type'] = 'Historical'

        # Combine the data
        df_combined = pd.concat([df_dart, df_predictions])

        # Add Symbol 
        df_combined['Symbol'] = SYMBOL


        # ------- Plot the forecast ------
        # generate plotly figure
        fig = go.Figure(data=[go.Scatter(x=df_dart.Date, y=df_dart.Close, name='actual'),
                        go.Scatter(x=df_pred.ds, y=df_pred.y, name='prediction')])
        fig.update_layout(title=f'Price Forecast Nbeats model for {SYMBOL}', 
                          xaxis_title='Date', yaxis_title='Price')
        fig.update_layout(template='plotly_dark')

        stored_figure = fig

        # ----- Save plot ------
        # save plot
        plot_dir = 'plots'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        fig.write_image(f'{plot_dir}/plot_nbeats_{SYMBOL}.png')

        # save as html
        fig.write_html(f'{plot_dir}/plot_nbeats_{SYMBOL}.html')


        # ----- Save predictions ------
        # save predictions
        data_dir = 'data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        df_combined.to_csv(f'{data_dir}/nbeats_forecast_{SYMBOL}.csv', index=False)

        if 'df_combined_all' not in locals():
            df_combined_all = df_combined
        else:
            df_combined_all = pd.concat([df_combined_all, df_combined])


    # Save the combined data
    df_combined_all.to_csv('data/combined_6_months_forecast.csv', index=False)
        

# -------------------------
# Run
# -------------------------

if __name__ == '__main__':
    main()


