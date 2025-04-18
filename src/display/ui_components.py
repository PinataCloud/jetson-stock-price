import pygame
from datetime import datetime
from ..config import Config

class StockInfoOverlay:
    """Displays NVIDIA stock information as an overlay"""
    
    def __init__(self, screen_width, screen_height):
        """Initialize the stock information overlay"""
        self.config = Config()
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Set up fonts
        self.large_font = pygame.font.Font(None, 48)
        self.medium_font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Stock data
        self.stock_data = None
        
        # Overlay properties
        self.opacity = self.config.stock['overlay_opacity']
        
        # Debug mode flag
        self.show_debug = False
    
    def update_stock_info(self, stock_data):
        """Update stock data for display
        
        Args:
            stock_data: Dictionary containing stock information
        """
        self.stock_data = stock_data
    
    def toggle_debug(self):
        """Toggle debug information display"""
        self.show_debug = not self.show_debug
    
    def draw(self, surface):
        """Draw the stock information overlay
        
        Args:
            surface: Pygame surface to draw on
        """
        if not self.stock_data:
            return
        
        # Create transparent overlay surface
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        
        # Draw main info panel
        self._draw_main_info(overlay)
        
        # Draw debug info if enabled
        if self.show_debug:
            self._draw_debug_info(overlay)
        
        # Draw the overlay onto the main surface
        surface.blit(overlay, (0, 0))
    
    def _draw_main_info(self, surface):
        """Draw the main stock information panel
        
        Args:
            surface: Pygame surface to draw on
        """
        # Calculate panel dimensions
        panel_width = 340
        panel_height = 120
        panel_x = self.screen_width - panel_width - 20
        panel_y = 20
        
        # Create semi-transparent panel
        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill((30, 30, 30, self.opacity))
        
        # Draw panel border
        pygame.draw.rect(panel, (60, 60, 60, self.opacity), panel.get_rect(), 2)
        
        # Draw company name and symbol
        company_text = self.medium_font.render(
            f"{self.stock_data['company_name']} ({self.stock_data['symbol']})",
            True, (255, 255, 255))
        panel.blit(company_text, (15, 15))
        
        # Draw current price
        price_color = self._get_price_color()
        price_text = self.large_font.render(
            f"${self.stock_data['current_price']:.2f}", 
            True, price_color)
        panel.blit(price_text, (15, 50))
        
        # Draw price change
        price_change = self.stock_data['price_change']
        price_change_pct = self.stock_data['price_change_pct']
        sign = "+" if price_change >= 0 else ""
        change_text = self.medium_font.render(
            f"{sign}{price_change:.2f} ({sign}{price_change_pct:.2f}%)", 
            True, price_color)
        panel.blit(change_text, (15, 85))
        
        # Draw last updated time
        update_time = self.small_font.render(
            f"Updated: {datetime.now().strftime('%H:%M:%S')}", 
            True, (200, 200, 200))
        panel.blit(update_time, (panel_width - update_time.get_width() - 15, 90))
        
        # Add panel to overlay
        surface.blit(panel, (panel_x, panel_y))
    
    def _draw_debug_info(self, surface):
        """Draw additional debug information
        
        Args:
            surface: Pygame surface to draw on
        """
        # Calculate panel dimensions
        panel_width = 400
        panel_height = 300
        panel_x = 20
        panel_y = self.screen_height - panel_height - 20
        
        # Create semi-transparent panel
        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill((20, 20, 20, self.opacity))
        
        # Draw panel border
        pygame.draw.rect(panel, (80, 80, 80, self.opacity), panel.get_rect(), 2)
        
        # Draw debug title
        debug_title = self.medium_font.render("Debug Information", True, (255, 255, 255))
        panel.blit(debug_title, (15, 15))
        
        # Add various debug info
        y_offset = 60
        line_height = 25
        
        # Add stock data details
        lines = [
            f"Symbol: {self.stock_data['symbol']}",
            f"Market Cap: ${self.stock_data.get('market_cap', 0) / 1e9:.2f}B",
            f"Volume: {self.stock_data.get('volume', 0):,}",
            f"Avg Volume: {self.stock_data.get('average_volume', 0):,}",
            f"Open: ${self.stock_data.get('open_price', 0):.2f}",
            f"High: ${self.stock_data.get('high_price', 0):.2f}",
            f"Low: ${self.stock_data.get('low_price', 0):.2f}",
            f"Data Range: {self.stock_data['chart_range'] if 'chart_range' in self.stock_data else self.config.stock['chart_range']}",
            f"Data Points: {len(self.stock_data.get('historical_data', [])) if 'historical_data' in self.stock_data else 'N/A'}",
            f"Fetch Time: {self.stock_data.get('fetch_time', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        
        for line in lines:
            text = self.small_font.render(line, True, (200, 200, 200))
            panel.blit(text, (15, y_offset))
            y_offset += line_height
        
        # Add panel to overlay
        surface.blit(panel, (panel_x, panel_y))
    
    def _get_price_color(self):
        """Get color based on price change direction
        
        Returns:
            Tuple (R,G,B) color
        """
        if not self.stock_data:
            return (255, 255, 255)  # Default white
            
        price_change = self.stock_data['price_change']
        
        if price_change > 0:
            return (0, 255, 0)  # Green for positive
        elif price_change < 0:
            return (255, 0, 0)  # Red for negative
        else:
            return (255, 255, 255)  # White for no change
