import pandas as pd
import yfinance as yf
import requests
import json
from datetime import datetime, timedelta
from ..config import Config

class StockFetcher:
    """Fetches NVIDIA stock data using Yahoo Finance API"""
    
    def __init__(self, symbol=None, debug=False):
        """Initialize the stock fetcher"""
        self.config = Config()
        self.debug = debug
        self.symbol = symbol or self.config.stock['symbol']
        self.last_data = None
        self.last_fetch_time = None
    
    def fetch_data(self):
        """Fetch stock data for NVIDIA"""
        try:
            # Get chart range from config
            chart_range = self.config.stock['chart_range']
            
            # Fetch historical data
            ticker = yf.Ticker(self.symbol)
            hist_data = ticker.history(period=chart_range)
            
            # Get the latest available data (might be different from today if market is closed)
            latest_date = hist_data.index[-1]
            
            # Get key stats
            current_price = hist_data['Close'].iloc[-1]
            open_price = hist_data['Open'].iloc[-1]
            high_price = hist_data['High'].iloc[-1]
            low_price = hist_data['Low'].iloc[-1]
            
            # Calculate price change
            if len(hist_data) > 1:
                prev_close = hist_data['Close'].iloc[-2]
                price_change = current_price - prev_close
                price_change_pct = (price_change / prev_close) * 100
            else:
                price_change = current_price - open_price
                price_change_pct = (price_change / open_price) * 100
            
            # Get company info
            info = ticker.info
            company_name = info.get('shortName', 'NVIDIA')
            
            # Create result data structure
            result = {
                'symbol': self.symbol,
                'company_name': company_name,
                'current_price': current_price,
                'open_price': open_price,
                'high_price': high_price,
                'low_price': low_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'latest_date': latest_date,
                'historical_data': hist_data,
                'fetch_time': datetime.now()
            }
            
            # Additional market info
            result['market_cap'] = info.get('marketCap', None)
            result['volume'] = info.get('volume', None)
            result['average_volume'] = info.get('averageVolume', None)
            
            # Store data for later reference
            self.last_data = result
            self.last_fetch_time = datetime.now()
            
            if self.debug:
                self._print_debug_info(result)
            
            return result
            
        except Exception as e:
            if self.debug:
                print(f"Error fetching stock data: {e}")
            
            # If we have last data, return it
            if self.last_data:
                print("Using cached data due to fetch error")
                return self.last_data
            
            return None
    
    def _print_debug_info(self, data):
        """Print debug information for the fetched data"""
        print(f"\n--- NVIDIA Stock Data ({data['fetch_time'].strftime('%Y-%m-%d %H:%M:%S')}) ---")
        print(f"Symbol: {data['symbol']}")
        print(f"Company: {data['company_name']}")
        print(f"Current Price: ${data['current_price']:.2f}")
        print(f"Change: ${data['price_change']:.2f} ({data['price_change_pct']:.2f}%)")
        print(f"Day Range: ${data['low_price']:.2f} - ${data['high_price']:.2f}")
        print(f"Market Cap: ${data['market_cap'] / 1e9:.2f}B")
        print(f"Volume: {data['volume']:,}")
        print("Historical Data Range:", data['historical_data'].index[0].strftime('%Y-%m-%d'), 
              "to", data['historical_data'].index[-1].strftime('%Y-%m-%d'))
        print(f"Data Points: {len(data['historical_data'])}")
        print("----------------------------------------------------\n")
