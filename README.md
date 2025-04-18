# StockVision - NVIDIA Stock Visualizer

A Python application that fetches NVIDIA stock data, visualizes it, and uses Stable Diffusion with ControlNet to render the visualization in a Studio Ghibli style. Developed for Jetson Orin Nano.

## Features

- **Stock Data Visualization**: Fetches real-time NVIDIA stock data using Yahoo Finance API
- **AI-Generated Backgrounds**: Uses Stable Diffusion with ControlNet to generate Studio Ghibli style visualizations
- **Automatic Updates**: Refreshes data and visualization every 30 minutes
- **Hardware Acceleration**: Optimized for NVIDIA Jetson Orin Nano using CUDA
- **Smooth Transitions**: Clean animations between visualization updates
- **Configurable Settings**: Extensive configuration options via YAML

## Requirements

- Python 3.8 or higher
- NVIDIA Jetson Orin Nano (or any CUDA-capable GPU)
- Dependencies listed in requirements.txt

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stockvision.git
   cd stockvision
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `models` directory and download required models:
   ```bash
   mkdir -p models
   ```

   You can download pre-trained Stable Diffusion checkpoints from Hugging Face.
   Recommended models:
   - [revAnimated_v2Rebirth.safetensors](https://huggingface.co/stablediffusionapi/revAnimated-v122)
   - [abstractPhoto_abcevereMix.safetensors](https://huggingface.co/stablediffusionapi/abstract-photography-diffusion)

## Usage

Run the application:

```bash
python main.py
```

Optional command-line arguments:
- `--debug`: Enable debug mode (saves debug images and shows verbose output)
- `--windowed`: Run in windowed mode instead of fullscreen

## Configuration

All settings can be customized in `config.yaml`. Important settings include:

- `display`: Screen resolution and FPS settings
- `render`: Image generation resolution and model settings
- `stock`: Stock symbol, update interval, and chart appearance
- `prompts`: Style controls for Studio Ghibli aesthetics

For machine-specific overrides (like local model paths), create a `local_config.yaml` file.

## Key Components

- **Stock Data**: `StockFetcher` retrieves NVIDIA stock data and `ChartRenderer` creates visualizations
- **Diffusion System**: `DiffusionPipeline` generates images with Stable Diffusion and ControlNet
- **Display System**: `SurfaceManager` handles visualization and `StockInfoOverlay` shows data

## Controls

- **ESC**: Exit application
- **F Key**: Toggle debug information display
- **R Key**: Force refresh visualization

## Acknowledgements

This project was inspired by the ClockRoss AI-Powered Analog Clock and uses several open-source libraries including:

- [Diffusers](https://github.com/huggingface/diffusers) - For Stable Diffusion and ControlNet
- [Pygame](https://www.pygame.org/) - For display and rendering
- [yfinance](https://github.com/ranaroussi/yfinance) - For stock data
- [Matplotlib](https://matplotlib.org/) - For chart rendering

## License

MIT License
