import pygame
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import io
from ..config import Config
from ..utils.image_utils import save_debug_image

class ChartRenderer:
    """Renders NVIDIA stock data as a chart surface for visualization"""
    
    def __init__(self, width, height, debug=False):
        """Initialize the chart renderer"""
        self.config = Config()
        self.debug = debug
        self.width = width
        self.height = height
        
        # Get chart colors from config
        self.chart_colors = self.config.stock['chart_colors']
        
        # Create a surface for rendering
        self.surface = pygame.Surface((width, height))
    
    def render_chart_array(self, stock_data):
        """Render the stock chart and return as numpy array
        
        This version doesn't use pygame directly to avoid threading issues
        """
        if not stock_data or 'historical_data' not in stock_data:
            # Return blank array if no data
            return np.ones((self.height, self.width, 3), dtype=np.uint8) * np.array(self.chart_colors['background'], dtype=np.uint8)
        
        # Get the historical data
        hist_data = stock_data['historical_data']
        
        # Create matplotlib figure
        plt.figure(figsize=(self.width/100, self.height/100), dpi=100)
        ax = plt.gca()
        
        # Plot based on chart type
        chart_type = self.config.stock['chart_type']
        
        if chart_type == 'candle':
            # Create OHLC chart
            self._plot_candlestick(ax, hist_data)
        else:
            # Create line chart (default)
            self._plot_line_chart(ax, hist_data)
        
        # Customize the plot
        self._customize_plot(ax, hist_data)
        
        # Add price and change annotation
        current_price = stock_data['current_price']
        price_change = stock_data['price_change']
        price_change_pct = stock_data['price_change_pct']
        
        # Determine color based on price change
        if price_change > 0:
            color = self._rgb_to_hex(self.chart_colors['up'])
        elif price_change < 0:
            color = self._rgb_to_hex(self.chart_colors['down'])
        else:
            color = self._rgb_to_hex(self.chart_colors['neutral'])
        
        # Add price and change annotations
        plt.annotate(
            f'${current_price:.2f}',
            xy=(0.98, 0.95),
            xycoords='axes fraction',
            fontsize=14,
            fontweight='bold',
            color=color,
            ha='right',
            va='top'
        )
        
        plt.annotate(
            f'{"+" if price_change >= 0 else ""}{price_change:.2f} ({price_change_pct:.2f}%)',
            xy=(0.98, 0.89),
            xycoords='axes fraction',
            fontsize=10,
            color=color,
            ha='right',
            va='top'
        )
        
        # Add current date/time
        plt.annotate(
            datetime.now().strftime('%Y-%m-%d %H:%M'),
            xy=(0.02, 0.02),
            xycoords='axes fraction',
            fontsize=8,
            color=self._rgb_to_hex(self.chart_colors['text']),
            ha='left',
            va='bottom',
            alpha=0.7
        )
        
        # Add company name and symbol
        plt.annotate(
            f"{stock_data['company_name']} ({stock_data['symbol']})",
            xy=(0.02, 0.95),
            xycoords='axes fraction',
            fontsize=12,
            fontweight='bold',
            color=self._rgb_to_hex(self.chart_colors['text']),
            ha='left',
            va='top'
        )
        
        # Save the plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, 
                   facecolor=self._rgb_to_hex(self.chart_colors['background']))
        buf.seek(0)
        
        # Convert buffer to numpy array directly
        from PIL import Image
        chart_image = Image.open(buf)
        chart_array = np.array(chart_image)
        
        plt.close()
        buf.close()
        
        # Resize array to match target dimensions if needed
        if chart_array.shape[0] != self.height or chart_array.shape[1] != self.width:
            from PIL import Image
            img = Image.fromarray(chart_array)
            img = img.resize((self.width, self.height), Image.LANCZOS)
            chart_array = np.array(img)
        
        if self.debug:
            # Save debug image
            Image.fromarray(chart_array).save(f"debug/chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        return chart_array
    
    def render_chart(self, stock_data):
        """Render the stock chart as a Pygame surface
        
        This should only be called from the main thread
        """
        # Get the array and convert to pygame surface
        array = self.render_chart_array(stock_data)
        
        # Convert to pygame surface - MUST be done on main thread
        surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        
        # Store the surface
        self.surface = surface
        
        return surface
    
    def _plot_line_chart(self, ax, hist_data):
        """Plot a line chart of closing prices"""
        # Get the closing prices
        closes = hist_data['Close']
        
        # Determine color based on price direction
        start_price = closes.iloc[0]
        end_price = closes.iloc[-1]
        color = self._rgb_to_hex(
            self.chart_colors['up'] if end_price >= start_price else self.chart_colors['down']
        )
        
        # Plot the line chart
        ax.plot(closes.index, closes, color=color, linewidth=2)
        
        # Add a marker for the most recent price
        ax.plot(closes.index[-1], closes.iloc[-1], 'o', 
               color=color, markersize=6)
    
    def _plot_candlestick(self, ax, hist_data):
        """Plot a candlestick chart"""
        # Iterate through data and plot each candle
        for i in range(len(hist_data)):
            # Get data for this candle
            date = hist_data.index[i]
            open_price = hist_data['Open'].iloc[i]
            close_price = hist_data['Close'].iloc[i]
            high_price = hist_data['High'].iloc[i]
            low_price = hist_data['Low'].iloc[i]
            
            # Determine candle color based on price direction
            if close_price >= open_price:
                color = self._rgb_to_hex(self.chart_colors['up'])
                candle_color = color
            else:
                color = self._rgb_to_hex(self.chart_colors['down'])
                candle_color = color
            
            # Plot high-low line (wick)
            ax.plot([date, date], [low_price, high_price], color=color, linewidth=1)
            
            # Plot candle body
            rect_height = max(0.001, abs(close_price - open_price))  # Ensure visible height
            rect_bottom = min(close_price, open_price)
            
            # Calculate width of candle (using 0.7 day width)
            width = 0.7
            
            # Plot rectangle for candle body
            rect = plt.Rectangle(
                (mdates.date2num(date) - width/2, rect_bottom),
                width, rect_height, 
                facecolor=candle_color, 
                edgecolor=color,
                alpha=0.8
            )
            ax.add_patch(rect)
    
    def _customize_plot(self, ax, hist_data):
        """Customize the plot appearance"""
        # Set background color
        ax.set_facecolor(self._rgb_to_hex(self.chart_colors['background']))
        
        # Customize grid
        ax.grid(True, linestyle='--', alpha=0.3, color=self._rgb_to_hex(self.chart_colors['grid']))
        
        # Customize axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(self._rgb_to_hex(self.chart_colors['grid']))
        ax.spines['left'].set_color(self._rgb_to_hex(self.chart_colors['grid']))
        
        # Set text colors
        ax.tick_params(axis='x', colors=self._rgb_to_hex(self.chart_colors['text']))
        ax.tick_params(axis='y', colors=self._rgb_to_hex(self.chart_colors['text']))
        
        # Format date on x-axis based on time frame
        timespan = hist_data.index[-1] - hist_data.index[0]
        
        if timespan <= timedelta(days=7):
            # For short timeframes, show detailed dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.xticks(rotation=45)
        elif timespan <= timedelta(days=180):
            # For medium timeframes, show month-day
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            plt.xticks(rotation=45)
        else:
            # For long timeframes, show month-year
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.xticks(rotation=45)
        
        # Format y-axis with dollar signs
        ax.yaxis.set_major_formatter('${x:.0f}')
        
        # Add minor gridlines
        ax.grid(True, which='minor', linestyle=':', alpha=0.2, color=self._rgb_to_hex(self.chart_colors['grid']))
        ax.minorticks_on()
        
        # Remove labels
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Adjust margins
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
    
    def _rgb_to_hex(self, rgb):
        """Convert RGB color to hex format for matplotlib"""
        return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])
