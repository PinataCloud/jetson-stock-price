import torch
import gc
import time
import os
from PIL import Image
import numpy as np
from diffusers import StableDiffusionPipeline
from ..config import Config
from ..utils.device_utils import get_best_device
from ..utils.image_utils import save_debug_image

class SimplifiedDiffusionPipeline:
    """Simplified Stable Diffusion pipeline that mimics ClockRoss's approach"""
    
    def __init__(self, debug=False):
        self.config = Config()
        self.debug = debug
        self.device = get_best_device()
        self.pipe = None
        self.is_loading = False
        
        if self.debug:
            print(f"Using device: {self.device}")
        
        # Initialize pipeline
        self._initialize_pipeline()
    
    def _empty_cache(self):
        """Properly clean up Python and device memory"""
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            # MPS doesn't have an explicit cache clearing mechanism
            # but we can force a sync point
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.synchronize()
        if self.debug:
            print("Memory cache cleared")

    def _initialize_pipeline(self):
        """Initialize a simplified Stable Diffusion pipeline"""
        if self.debug:
            print("Initializing simplified Stable Diffusion pipeline...")
        
        try:
            # Create models directory if it doesn't exist
            os.makedirs("models", exist_ok=True)
            
            # Try loading Ghibli-Diffusion model
            try:
                print("Loading Ghibli-Diffusion model...")
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    "nitrosocke/Ghibli-Diffusion",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None
                ).to(self.device)
                
            except Exception as e:
                print(f"Failed to load Ghibli model: {e}")
                print("Falling back to standard Stable Diffusion 1.5...")
                
                # Fall back to standard SD 1.5
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None
                ).to(self.device)
            
            # Set lower precision for faster CPU performance
            if self.device == "cpu":
                print("Setting to float32 for CPU...")
                self.pipe.to(torch.float32)
                # Use attention slicing for better memory usage on CPU
                self.pipe.enable_attention_slicing()
                
            # Enable memory efficient attention for CUDA
            if self.device == "cuda":
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    print("Enabled xformers for efficient memory usage")
                except:
                    try:
                        self.pipe.enable_attention_slicing()
                        print("Enabled attention slicing for lower memory usage")
                    except:
                        print("Using default attention")
            
            # Get or update generation settings
            gen_config = self.config.render['generation']
            self.pipe.scheduler.config.beta_schedule = "scaled_linear"
            self.inference_steps = gen_config.get('num_inference_steps', 10)
            self.guidance_scale = gen_config.get('guidance_scale', 7.5)
            
            print("Pipeline initialized successfully")
            
        except Exception as e:
            print(f"Error initializing pipeline: {e}")
            raise

    def generate(self, image, prompt):
        """Generate an image using Stable Diffusion
        
        Args:
            image: PIL Image to use as reference (not used for conditioning)
            prompt: Text prompt for generation
            
        Returns:
            Tuple of (PIL Image, seed)
        """
        if self.pipe is None:
            raise RuntimeError("Pipeline not initialized")
        
        # Create a fixed size for consistent output
        width = self.config.render['width']
        height = self.config.render['height']
        
        if self.debug:
            print(f"Generating image with dimensions {width}x{height}")
            print(f"Prompt: {prompt}")
            if image:
                image.save(f"debug/input_image_{time.strftime('%Y%m%d_%H%M%S')}.png")
        
        try:
            # Generate random seed
            seed = torch.randint(0, 2**32, (1,)).item()
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Get negative prompt
            negative_prompt = self.config.prompts['negative_prompt']
            
            # Choose inference steps based on device
            steps = self.inference_steps
            if self.device == "cpu":
                # Use fewer steps for CPU
                steps = min(steps, 20)
                if self.debug:
                    print(f"Using {steps} inference steps for CPU")
            
            # Generate image
            start_time = time.time()
            
            # Generate image - simplified mode
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=self.guidance_scale,
                generator=generator,
                height=height,
                width=width
            )
            
            end_time = time.time()
            
            if self.debug:
                print(f"Image generation completed in {end_time - start_time:.2f} seconds")
                # Save image for debugging
                generated_image = result.images[0]
                generated_image.save(f"debug/generated_{time.strftime('%Y%m%d_%H%M%S')}.png")
            
            return result.images[0], seed
            
        except Exception as e:
            if self.debug:
                print(f"Error generating image: {e}")
            
            # Create a fallback image showing the error
            fallback_image = Image.new('RGB', (width, height), (30, 30, 30))
            return fallback_image, 0
