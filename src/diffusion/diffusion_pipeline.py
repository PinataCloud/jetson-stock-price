import torch
import gc
import time
import threading
import os
from pathlib import Path
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DiffusionPipeline as HFDiffusionPipeline
from PIL import Image
import numpy as np
import pygame
from ..config import Config
from ..utils.device_utils import get_best_device
from ..utils.image_utils import save_debug_image

class DiffusionPipeline:
    """Wraps Stable Diffusion with ControlNet for chart-guided image generation"""
    
    def __init__(self, debug=False):
        self.config = Config()
        self.debug = debug
        self.device = get_best_device()
        self.pipe = None
        self.controlnet_pipe = None
        self.is_loading = False
        self.reload_complete_callback = None
        self.reload_error_callback = None
        
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
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
        if self.debug:
            print("Memory cache cleared")

    def _cleanup_pipeline(self):
        """Clean up the existing pipeline to free GPU memory"""
        if hasattr(self, 'pipe') and self.pipe is not None:
            try:
                del self.pipe
                self.pipe = None
                self._empty_cache()
                time.sleep(1)  # Small delay to ensure cleanup
            except Exception as e:
                if self.debug:
                    print(f"Error cleaning up pipeline: {e}")
        
        if hasattr(self, 'controlnet_pipe') and self.controlnet_pipe is not None:
            try:
                del self.controlnet_pipe
                self.controlnet_pipe = None
                self._empty_cache()
            except Exception as e:
                if self.debug:
                    print(f"Error cleaning up controlnet pipeline: {e}")

    def _initialize_pipeline(self):
        """Initialize the Stable Diffusion pipeline with ControlNet"""
        if self.debug:
            print("Initializing Stable Diffusion pipelines...")
        
        try:
            # Load the Ghibli-Diffusion model directly
            print("Loading Ghibli-Diffusion model...")
            self.pipe = HFDiffusionPipeline.from_pretrained(
                "nitrosocke/Ghibli-Diffusion",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None
            ).to(self.device)
            
            # Enable memory efficient attention for CUDA
            if self.device == "cuda":
                self.pipe.enable_xformers_memory_efficient_attention()
            
            # Now load ControlNet model for chart-guided generation
            print("Loading ControlNet model...")
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11f1e_sd15_tile",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            # Use StableDiffusionControlNetPipeline as a second pipeline for chart-guided images
            print("Setting up ControlNet pipeline...")
            self.controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "nitrosocke/Ghibli-Diffusion",  # Use the same Ghibli model with controlnet
                controlnet=controlnet,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None
            ).to(self.device)
            
            # Enable memory efficient attention for ControlNet pipeline
            if self.device == "cuda":
                self.controlnet_pipe.enable_xformers_memory_efficient_attention()
            
            print("Pipelines initialized successfully")
        except Exception as e:
            print(f"Error initializing pipelines: {e}")
            raise

    def reload(self, complete_callback=None, error_callback=None):
        """Reload the pipeline with new configuration in a separate thread"""
        self.is_loading = True
        def wrapped_callback():
            self.is_loading = False
            if complete_callback:
                complete_callback()
        self.reload_complete_callback = wrapped_callback
        self.reload_error_callback = error_callback
        reload_thread = threading.Thread(target=self._do_reload_pipeline)
        reload_thread.daemon = True
        reload_thread.start()
    
    def _do_reload_pipeline(self):
        """Internal method to handle the actual pipeline reload"""
        try:
            if self.debug:
                print("Starting pipeline reload...")
            
            # Clean up existing pipeline
            if self.debug:
                print("Cleaning up old pipeline...")
            self._cleanup_pipeline()
            
            if self.debug:
                print("Loading new pipeline...")
            self._initialize_pipeline()
            
            if self.debug:
                print("Pipeline reload complete")
            
            # Notify completion if callback is set
            if self.reload_complete_callback:
                self.reload_complete_callback()
        except Exception as e:
            if self.reload_error_callback:
                self.reload_error_callback(e)
            if self.debug:
                print(f"Error reloading pipeline: {e}")

    def generate(self, chart_surface, prompt):
        """Generate a Studio Ghibli style image using the chart as conditioning input"""
        if self.pipe is None or self.controlnet_pipe is None:
            raise RuntimeError("Pipeline not initialized")
        
        if self.debug:
            print(f"Generating image with prompt: {prompt}")
            save_debug_image(chart_surface, "control_input")
        
        try:
            # Convert pygame surface to PIL Image for ControlNet input
            chart_array = pygame.surfarray.array3d(chart_surface)
            # Pygame uses a different axis order, so we need to transpose
            chart_array = chart_array.transpose(1, 0, 2)
            chart_image = Image.fromarray(chart_array)
            
            # Get generation settings from config
            gen_config = self.config.render['generation']
            
            # Generate random seed
            seed = torch.randint(0, 2**32, (1,)).item()
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Get negative prompt
            negative_prompt = self.config.prompts['negative_prompt']
            
            # Use controlnet to generate an image guided by the chart
            start_time = time.time()
            
            # Decide randomly whether to use controlnet or pure diffusion
            use_controlnet = np.random.random() < 0.7  # 70% chance to use controlnet
            
            if use_controlnet and chart_surface is not None:
                # Use ControlNet pipeline for chart-guided generation
                result = self.controlnet_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=chart_image,
                    num_inference_steps=gen_config['num_inference_steps'],
                    guidance_scale=gen_config['guidance_scale'],
                    controlnet_conditioning_scale=gen_config['controlnet_conditioning_scale'],
                    generator=generator
                )
            else:
                # Use regular Ghibli diffusion without chart guidance
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=gen_config['num_inference_steps'],
                    guidance_scale=gen_config['guidance_scale'],
                    generator=generator
                )
            
            end_time = time.time()
            
            if self.debug:
                print(f"Image generation completed in {end_time - start_time:.2f} seconds")
                pipeline_type = "ControlNet" if use_controlnet else "Standard"
                print(f"Using {pipeline_type} pipeline")
                save_debug_image(result.images[0], "generated")
            
            return result.images[0], seed
            
        except Exception as e:
            if self.debug:
                print(f"Error generating image: {e}")
            raise
