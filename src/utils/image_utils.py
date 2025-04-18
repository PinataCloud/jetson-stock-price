import pygame
from PIL import Image
import cv2
import numpy as np
import os
from datetime import datetime
import time

def save_debug_image(image, prefix):
    """Save a debug image with timestamp.
    
    Args:
        image: Either a pygame Surface, PIL Image, or numpy array
        prefix: String prefix for the filename (e.g., 'chart' or 'generated')
    """
    # Create debug directory if it doesn't exist
    os.makedirs('debug', exist_ok=True)
    
    # Generate debug filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    debug_filename = f"debug/debug_{prefix}_{timestamp}.png"
    
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    elif isinstance(image, pygame.Surface):
        # Convert pygame surface to PIL image
        array = pygame.surfarray.array3d(image)
        image = Image.fromarray(array.swapaxes(0, 1))
    
    # Save the image
    image.save(debug_filename)
    print(f"Saved {prefix} debug image to {debug_filename}")

def pygame_to_pil(surface):
    """Convert a pygame surface to PIL Image
    
    Args:
        surface: Pygame surface
        
    Returns:
        PIL Image
    """
    array = pygame.surfarray.array3d(surface)
    # Transpose to correct orientation (Pygame uses different axis order)
    array = array.transpose(1, 0, 2)
    return Image.fromarray(array)

def pil_to_pygame(pil_image):
    """Convert PIL Image to pygame surface
    
    Args:
        pil_image: PIL Image
        
    Returns:
        Pygame surface
    """
    array = np.array(pil_image)
    return pygame.surfarray.make_surface(array.swapaxes(0, 1))

def cv2_to_pil(cv_image):
    """Convert OpenCV image to PIL Image
    
    Args:
        cv_image: OpenCV image (numpy array in BGR format)
        
    Returns:
        PIL Image (in RGB format)
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)

def pil_to_cv2(pil_image):
    """Convert PIL Image to OpenCV format
    
    Args:
        pil_image: PIL Image
        
    Returns:
        OpenCV image (numpy array in BGR format)
    """
    # Convert PIL to numpy array (RGB)
    rgb_array = np.array(pil_image)
    # Convert RGB to BGR
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

def resize_image(image, width, height, mode='pil'):
    """Resize an image to the specified dimensions
    
    Args:
        image: PIL Image or pygame Surface
        width: Target width
        height: Target height
        mode: 'pil' or 'pygame'
        
    Returns:
        Resized image in the same format as input
    """
    if mode == 'pil':
        if isinstance(image, pygame.Surface):
            # Convert pygame -> PIL, resize, convert back
            pil_img = pygame_to_pil(image)
            resized = pil_img.resize((width, height), Image.Resampling.LANCZOS)
            return resized
        else:
            # Resize PIL image directly
            return image.resize((width, height), Image.Resampling.LANCZOS)
    else:  # pygame mode
        if isinstance(image, pygame.Surface):
            # Resize pygame surface directly
            return pygame.transform.smoothscale(image, (width, height))
        else:
            # Convert PIL -> pygame, resize
            pygame_img = pil_to_pygame(image)
            return pygame.transform.smoothscale(pygame_img, (width, height))
