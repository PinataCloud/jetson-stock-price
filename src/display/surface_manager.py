import pygame
import numpy as np
import os
import json
import time
from datetime import datetime
from PIL import Image
from ..config import Config
from ..utils.image_utils import save_debug_image

class SurfaceManager:
    """Manages the display surfaces and transitions for the visualization"""
    
    def __init__(self, display_width, display_height, render_width, render_height, debug=False):
        """Initialize the surface manager"""
        self.config = Config()
        self.display_width = display_width
        self.display_height = display_height
        self.render_width = render_width
        self.render_height = render_height
        self.debug = debug
        
        # Initialize surfaces
        self.background_surface = None
        self.previous_background = None
        self.transition_progress = 1.0  # Start fully transitioned
        
        # Create snapshots directory
        self.snapshots_dir = "snapshots"
        os.makedirs(self.snapshots_dir, exist_ok=True)
        
        # Metadata
        self.last_render_request = None
        
        # Queue for background updates from other threads
        self.pending_background = None
    
    def queue_background_update(self, image_data):
        """Queue a background update from another thread
        
        Args:
            image_data: PIL Image
        """
        # Just store the PIL image - we'll convert to pygame surface on the main thread
        self.pending_background = image_data
        if self.debug:
            print("Background update queued")
    
    def apply_pending_updates(self):
        """Apply any pending updates (should be called from main thread)"""
        if self.pending_background:
            self.update_background(self.pending_background)
            self.pending_background = None
            if self.debug:
                print("Pending background update applied")
    
    def update_background(self, image_data):
        """Update the background surface with new image data
        
        Args:
            image_data: PIL Image
        """
        # Save previous background for transitions
        if self.background_surface:
            self.previous_background = self.background_surface
            self.transition_progress = 0.0
        
        # Convert PIL Image to pygame surface - MUST be done on main thread
        array = np.array(image_data)
        self.background_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        
        if self.debug:
            save_debug_image(image_data, "background")
    
    def get_display_background(self):
        """Get the current background surface, handling transitions
        
        Returns:
            Pygame surface for display
        """
        if not self.background_surface:
            return None
        
        # Handle transitions between backgrounds
        if self.previous_background and self.transition_progress < 1.0:
            # Update transition progress based on config duration
            fps = self.config.display['fps']
            duration = self.config.animation['transition_duration']
            self.transition_progress += 1.0 / (fps * duration)
            self.transition_progress = min(1.0, self.transition_progress)
            
            # Create transition surface
            transition = pygame.Surface((self.display_width, self.display_height))
            
            # Draw previous background
            prev_scaled = pygame.transform.smoothscale(
                self.previous_background, 
                (self.display_width, self.display_height)
            )
            transition.blit(prev_scaled, (0, 0))
            
            # Draw current background with alpha
            current_scaled = pygame.transform.smoothscale(
                self.background_surface, 
                (self.display_width, self.display_height)
            )
            
            # Set alpha for current background
            current_scaled.set_alpha(int(255 * self.transition_progress))
            
            # Blend backgrounds
            transition.blit(current_scaled, (0, 0))
            return transition
        else:
            # Return scaled background
            return pygame.transform.smoothscale(
                self.background_surface, 
                (self.display_width, self.display_height)
            )
    
    def update_render_request(self, render_request):
        """Update the last render request metadata
        
        Args:
            render_request: Dictionary with metadata about the render
        """
        self.last_render_request = render_request
    
    def save_snapshot(self, stock_data=None):
        """Save current visualization to snapshot files
        
        Args:
            stock_data: Optional stock data to include in metadata
        """
        if not self.background_surface:
            return
            
        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save background image
        pygame.image.save(
            self.background_surface, 
            f"{self.snapshots_dir}/{timestamp}_background.png"
        )
        
        # Create metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "render_width": self.render_width,
            "render_height": self.render_height,
            "display_width": self.display_width,
            "display_height": self.display_height,
        }
        
        # Add stock data if available
        if stock_data:
            metadata["stock"] = {
                "symbol": stock_data.get("symbol"),
                "company_name": stock_data.get("company_name"),
                "current_price": float(stock_data.get("current_price", 0)),
                "price_change": float(stock_data.get("price_change", 0)),
                "price_change_pct": float(stock_data.get("price_change_pct", 0)),
                "fetch_time": stock_data.get("fetch_time", datetime.now()).isoformat()
            }
        
        # Add render request if available
        if self.last_render_request:
            metadata["render"] = self.last_render_request
        
        # Save metadata
        with open(f"{self.snapshots_dir}/{timestamp}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        if self.debug:
            print(f"Snapshot saved to {self.snapshots_dir}/{timestamp}_*.png/json")
        
        return timestamp
