import pygame
import argparse
import threading
import time
import os
from datetime import datetime
from src.stock.stock_fetcher import StockFetcher
from src.stock.chart_renderer import ChartRenderer
from src.diffusion.diffusion_pipeline import DiffusionPipeline
from src.diffusion.prompt_generator import PromptGenerator
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

def parse_args():
    parser = argparse.ArgumentParser(description='NVIDIA Stock Visualizer with Studio Ghibli Style')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (save debug images and verbose output)')
    parser.add_argument('--windowed', action='store_true', help='Run in windowed mode instead of fullscreen')
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
    splash = pygame.image.load('splash.png')
    splash = pygame.transform.smoothscale(splash, (display_width, display_height))
    screen.blit(splash, (0, 0))
    pygame.display.flip()
    
    # Initialize components
    clock = pygame.time.Clock()
    surface_manager = SurfaceManager(display_width, display_height, RENDER_WIDTH, RENDER_HEIGHT, debug=debug)
    
    # Create stock fetcher and chart renderer
    stock_fetcher = StockFetcher(symbol="NVDA", debug=debug)
    chart_renderer = ChartRenderer(RENDER_WIDTH, RENDER_HEIGHT, debug=debug)
    
    # Create Stable Diffusion pipeline and prompt generator
    prompt_generator = PromptGenerator()
    diffusion_pipeline = DiffusionPipeline(debug=debug)
    
    # Create stock info overlay
    stock_overlay = StockInfoOverlay(display_width, display_height)
    
    # Shared state variables
    last_update_time = 0
    update_lock = threading.Lock()
    is_updating = False
    
    def update_visualization():
        nonlocal is_updating
        
        # Set updating flag
        with update_lock:
            is_updating = True
        
        try:
            if debug:
                print(f"Starting visualization update at {datetime.now().strftime('%H:%M:%S')}")
            
            # Fetch latest stock data
            stock_data = stock_fetcher.fetch_data()
            
            if stock_data:
                # Render chart
                chart_surface = chart_renderer.render_chart(stock_data)
                
                # Generate Studio Ghibli style prompt
                prompt = prompt_generator.generate_prompt(stock_data)
                
                if debug:
                    print(f"Generated prompt: {prompt}")
                
                # Generate image using Stable Diffusion
                image, seed = diffusion_pipeline.generate(chart_surface, prompt)
                
                # Update display
                surface_manager.update_background(image)
                stock_overlay.update_stock_info(stock_data)
                
                if debug:
                    print(f"Visualization updated successfully at {datetime.now().strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"Error updating visualization: {e}")
        finally:
            # Clear updating flag
            with update_lock:
                is_updating = False
    
    # Initial update
    update_thread = threading.Thread(target=update_visualization)
    update_thread.daemon = True
    update_thread.start()
    
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
                    if not is_updating:
                        update_thread = threading.Thread(target=update_visualization)
                        update_thread.daemon = True
                        update_thread.start()
        
        # Check if it's time for an update
        current_time = time.time()
        with update_lock:
            should_update = not is_updating and (current_time - last_update_time) >= UPDATE_INTERVAL
        
        if should_update:
            last_update_time = current_time
            update_thread = threading.Thread(target=update_visualization)
            update_thread.daemon = True
            update_thread.start()
        
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
