import pygame
import argparse
import time
import os
from datetime import datetime
from src.stock.stock_fetcher import StockFetcher
from src.stock.chart_renderer import ChartRenderer
from src.diffusion.simplified_pipeline import SimplifiedDiffusionPipeline
from src.diffusion.prompt_generator import PromptGenerator
from src.display.surface_manager import SurfaceManager
from src.display.ui_components import StockInfoOverlay
from src.config import Config
from PIL import Image
import numpy as np

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
    
    # Font for loading messages
    font = pygame.font.Font(None, 36)
    
    # State for update timing
    last_update_time = 0
    is_updating = False
    update_stage = "idle"  # Track what we're currently doing
    
    # Initialize components
    clock = pygame.time.Clock()
    surface_manager = SurfaceManager(display_width, display_height, RENDER_WIDTH, RENDER_HEIGHT, debug=debug)
    
    # Create stock fetcher and chart renderer
    stock_fetcher = StockFetcher(symbol="NVDA", debug=debug)
    chart_renderer = ChartRenderer(RENDER_WIDTH, RENDER_HEIGHT, debug=debug)
    
    # Create stock info overlay
    stock_overlay = StockInfoOverlay(display_width, display_height)
    
    # Create AI components only if not disabled
    if not args.no_ai:
        prompt_generator = PromptGenerator()
        diffusion_pipeline = SimplifiedDiffusionPipeline(debug=debug)
    
    # Variables for staged update process
    stock_data = None
    chart_surface = None
    prompt = None
    generated_image = None
    
    def show_loading_message(message):
        """Display a loading message"""
        # Preserve a portion of the existing screen (so it's not a full-screen flash)
        overlay = pygame.Surface((display_width, 50), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))  # Semi-transparent black
        
        # Render text
        loading_text = font.render(message, True, (255, 255, 255))
        overlay.blit(loading_text, (overlay.get_width()//2 - loading_text.get_width()//2, 
                                   overlay.get_height()//2 - loading_text.get_height()//2))
        
        # Draw on screen at the bottom
        screen.blit(overlay, (0, display_height - 50))
        pygame.display.update(pygame.Rect(0, display_height - 50, display_width, 50))
    
    # Initial loading screen
    screen.fill(BACKGROUND_COLOR)
    loading_text = font.render("Loading NVIDIA Stock Visualizer...", True, (255, 255, 255))
    screen.blit(loading_text, (display_width//2 - loading_text.get_width()//2, display_height//2))
    pygame.display.flip()
    
    running = True
    while running:
        # Process events
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
                        is_updating = True
                        update_stage = "fetch_data"
                        last_update_time = time.time()
                elif event.key == pygame.K_n and not is_updating:
                    # Toggle AI generation on/off
                    args.no_ai = not args.no_ai
                    if args.no_ai:
                        show_loading_message("AI generation disabled")
                    else:
                        show_loading_message("AI generation enabled")
                        # Initialize AI if it was disabled before
                        if not 'prompt_generator' in locals():
                            prompt_generator = PromptGenerator()
                            diffusion_pipeline = SimplifiedDiffusionPipeline(debug=debug)
        
        # Check if it's time for an update
        current_time = time.time()
        if not is_updating and (current_time - last_update_time) >= UPDATE_INTERVAL:
            is_updating = True
            update_stage = "fetch_data"
            last_update_time = current_time
        
        # If we're updating, process the current stage
        if is_updating:
            try:
                # Staged update process - each frame we check where we are in the process
                if update_stage == "fetch_data":
                    show_loading_message("Fetching NVIDIA stock data...")
                    stock_data = stock_fetcher.fetch_data()
                    update_stage = "render_chart" if stock_data else "done"
                
                elif update_stage == "render_chart":
                    show_loading_message("Rendering stock chart...")
                    chart_surface = chart_renderer.render_chart(stock_data)
                    stock_overlay.update_stock_info(stock_data)
                    
                    if args.no_ai:
                        # Use chart as background if AI is disabled
                        pygame_array = pygame.surfarray.array3d(chart_surface)
                        # Convert PyGame array to PIL Image
                        array = pygame.surfarray.array3d(chart_surface)
                        array = array.transpose(1, 0, 2)
                        chart_image = Image.fromarray(array)
                        surface_manager.update_background(chart_image)
                        update_stage = "done"  # Skip AI generation
                    else:
                        update_stage = "generate_prompt"
                
                elif update_stage == "generate_prompt":
                    show_loading_message("Generating Ghibli-style prompt...")
                    prompt = prompt_generator.generate_prompt(stock_data)
                    if debug:
                        print(f"Generated prompt: {prompt}")
                    update_stage = "generate_image"
                
                elif update_stage == "generate_image":
                    show_loading_message("Generating Ghibli-style visualization...")
                    generated_image, seed = diffusion_pipeline.generate(chart_surface, prompt)
                    surface_manager.update_background(generated_image)
                    update_stage = "done"
                
                elif update_stage == "done":
                    if debug:
                        print(f"Visualization updated successfully at {datetime.now().strftime('%H:%M:%S')}")
                    is_updating = False
                    update_stage = "idle"
            
            except Exception as e:
                print(f"Error in {update_stage}: {e}")
                is_updating = False
                update_stage = "idle"
        
        # Regular rendering
        if not is_updating or update_stage != "idle":  # Always render during updates
            # Clear screen
            screen.fill(BACKGROUND_COLOR)
            
            # Draw background
            bg_surface = surface_manager.get_display_background()
            if bg_surface:
                screen.blit(bg_surface, (0, 0))
            
            # Draw stock info overlay
            if stock_overlay:
                stock_overlay.draw(screen)
            
            pygame.display.flip()
        
        # Control frame rate
        clock.tick(config.display['fps'])

    pygame.quit()

if __name__ == "__main__":
    main()
