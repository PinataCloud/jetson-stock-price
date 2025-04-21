import pygame
import argparse
import threading
import time
import os
from datetime import datetime
from PIL import Image
import numpy as np
from src.stock.stock_fetcher import StockFetcher
from src.stock.chart_renderer import ChartRenderer
from src.display.surface_manager import SurfaceManager
from src.display.ui_components import StockInfoOverlay
from src.config import Config

# Set Hugging Face cache directories
os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), 'cache', 'hf')

# Create cache directories if they don't exist
os.makedirs(os.environ['HF_HOME'], exist_ok=True)

# Initialize Pygame
pygame.init()

# Get the display info and load config
display_info = pygame.display.Info()
config = Config()

# Get display settings
FULLSCREEN_WIDTH = display_info.current_w
FULLSCREEN_HEIGHT = display_info.current_h
WINDOWED_WIDTH = config.display['windowed_width']
WINDOWED_HEIGHT = config.display['windowed_height']
RENDER_WIDTH = config.render['width']
RENDER_HEIGHT = config.render['height']
BACKGROUND_COLOR = tuple(config.render['background_color'])
UPDATE_INTERVAL = config.stock['update_interval_minutes'] * 60  # Convert to seconds

class BackgroundUpdater:
    """Similar pattern to ClockRoss BackgroundUpdater"""
    def __init__(self, debug=False):
        self.config = Config()
        self.debug = debug
        self.surface_manager = None
        self.stock_overlay = None
        self.lock = threading.Lock()
        self.last_attempt = 0
        self.is_updating = False
        self.update_thread = None
        
        # Initialize stock components
        self.stock_fetcher = StockFetcher(symbol="NVDA", debug=debug)
        self.chart_renderer = ChartRenderer(RENDER_WIDTH, RENDER_HEIGHT, debug=debug)
        
        # Only load AI components if available and working
        self.diffusion_enabled = False
        try:
            from src.diffusion.simplified_pipeline import SimplifiedDiffusionPipeline
            from src.diffusion.prompt_generator import PromptGenerator
            self.prompt_generator = PromptGenerator()
            self.diffusion_pipeline = SimplifiedDiffusionPipeline(debug=debug)
            self.diffusion_enabled = True
            if self.debug:
                print("AI image generation enabled")
        except Exception as e:
            if self.debug:
                print(f"AI image generation disabled: {e}")
            self.diffusion_enabled = False
    
    def set_surface_manager(self, surface_manager):
        """Set the surface manager instance"""
        self.surface_manager = surface_manager
    
    def set_stock_overlay(self, stock_overlay):
        """Set the stock overlay instance"""
        self.stock_overlay = stock_overlay
    
    def _do_update(self):
        """Internal method that runs in a separate thread to update the visualization"""
        try:
            if self.debug:
                print(f"Starting visualization update at {datetime.now().strftime('%H:%M:%S')}")
            
            # Fetch latest stock data
            stock_data = self.stock_fetcher.fetch_data()
            
            if stock_data:
                # Render chart - do this in the thread as it doesn't use pygame directly
                chart_array = self.chart_renderer.render_chart_array(stock_data)
                
                # Update stock info (will be applied on main thread)
                if self.stock_overlay:
                    self.stock_overlay.update_stock_info(stock_data)
                
                # Generate image using AI if enabled
                if self.diffusion_enabled:
                    try:
                        # Convert numpy array to PIL Image for diffusion input
                        chart_image = Image.fromarray(chart_array)
                        
                        # Generate prompt
                        prompt = self.prompt_generator.generate_prompt(stock_data)
                        
                        if self.debug:
                            print(f"Generated prompt: {prompt}")
                        
                        # Generate image using Stable Diffusion
                        image, seed = self.diffusion_pipeline.generate(chart_image, prompt)
                        
                        # Update background in surface manager (will be applied on main thread)
                        if self.surface_manager:
                            self.surface_manager.queue_background_update(image)
                        
                        if self.debug:
                            print(f"AI visualization generated successfully")
                    except Exception as e:
                        print(f"Error in AI generation: {e}")
                        # Fall back to using the chart directly
                        if self.surface_manager:
                            self.surface_manager.queue_background_update(Image.fromarray(chart_array))
                else:
                    # Use chart directly if AI is disabled
                    if self.surface_manager:
                        self.surface_manager.queue_background_update(Image.fromarray(chart_array))
                
                if self.debug:
                    print(f"Visualization update prepared at {datetime.now().strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"Error updating visualization: {e}")
        finally:
            with self.lock:
                self.is_updating = False
                self.update_thread = None
    
    def update_background(self):
        """Start a background update if conditions are met"""
        current_time = time.time()
        with self.lock:
            # Don't update if we're already updating
            if self.is_updating or (current_time - self.last_attempt) < UPDATE_INTERVAL:
                return
                
            self.is_updating = True
            self.last_attempt = current_time
            
            # Create and start a new thread for the update
            self.update_thread = threading.Thread(
                target=self._do_update
            )
            self.update_thread.daemon = True  # Thread will be killed when main program exits
            self.update_thread.start()
    
    def should_update(self):
        """Check if it's time for a background update"""
        return time.time() - self.last_attempt >= UPDATE_INTERVAL

def parse_args():
    parser = argparse.ArgumentParser(description='NVIDIA Stock Visualizer with Studio Ghibli Style')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (save debug images and verbose output)')
    parser.add_argument('--windowed', action='store_true', help='Run in windowed mode instead of fullscreen')
    parser.add_argument('--no-ai', action='store_true', help='Disable AI image generation (use chart only)')
    return parser.parse_args()

def main():
    args = parse_args()
    debug = args.debug
    
    # Set up display
    if args.windowed:
        screen = pygame.display.set_mode((WINDOWED_WIDTH, WINDOWED_HEIGHT))
        display_width, display_height = WINDOWED_WIDTH, WINDOWED_HEIGHT
    else:
        screen = pygame.display.set_mode((FULLSCREEN_WIDTH, FULLSCREEN_HEIGHT), pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF)
        display_width, display_height = FULLSCREEN_WIDTH, FULLSCREEN_HEIGHT
        pygame.mouse.set_visible(False)  # Hide cursor only in fullscreen mode
    pygame.display.set_caption("NVIDIA Stock Visualizer")
    
    # Show splash screen
    # Create a simple splash screen
    font = pygame.font.Font(None, 48)
    splash_text = font.render("NVIDIA Stock Visualizer", True, (255, 255, 255))
    screen.fill(BACKGROUND_COLOR)
    screen.blit(splash_text, (display_width//2 - splash_text.get_width()//2, display_height//2 - 50))
    
    # Add loading text
    loading_font = pygame.font.Font(None, 24)
    loading_text = loading_font.render("Loading...", True, (200, 200, 200))
    screen.blit(loading_text, (display_width//2 - loading_text.get_width()//2, display_height//2 + 20))
    pygame.display.flip()
    
    # Initialize components
    clock = pygame.time.Clock()
    surface_manager = SurfaceManager(display_width, display_height, RENDER_WIDTH, RENDER_HEIGHT, debug=debug)
    
    # Create stock info overlay
    stock_overlay = StockInfoOverlay(display_width, display_height)
    
    # Create background updater
    background_updater = BackgroundUpdater(debug=debug)
    background_updater.set_surface_manager(surface_manager)
    background_updater.set_stock_overlay(stock_overlay)
    
    # Force initial update
    background_updater.update_background()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_f:
                    # Toggle debug info display
                    if debug:
                        stock_overlay.toggle_debug()
                elif event.key == pygame.K_r:
                    # Force refresh
                    background_updater.last_attempt = 0  # Reset timer to force update
        
        # Check if we should update
        if background_updater.should_update():
            background_updater.update_background()
        
        # Apply any pending updates from surface manager (this is safe to do on main thread)
        surface_manager.apply_pending_updates()
        
        # Clear screen
        screen.fill(BACKGROUND_COLOR)
        
        # Draw background
        bg_surface = surface_manager.get_display_background()
        if bg_surface:
            screen.blit(bg_surface, (0, 0))
        
        # Draw stock info overlay
        stock_overlay.draw(screen)
        
        pygame.display.flip()
        clock.tick(config.display['fps'])

    pygame.quit()

if __name__ == "__main__":
    main()
