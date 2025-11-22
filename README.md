# UltrastarKaraokeMaker

A powerful tool for automatically creating karaoke tracks from Ultrastar song folders. This tool separates vocals from instrumentals and creates karaoke-ready files with backing vocals at customizable volume levels.

## Features

- 🎵 **Automatic Audio Separation**: Uses the `mel_band_roformer_karaoke_aufr33_viperx_sdr_10` AI model to separate vocals from instrumentals
- 🎤 **Karaoke Track Generation**: Creates instrumental tracks with subtle backing vocals for better singing experience
- 📁 **Batch Processing**: Process multiple song folders at once with offset and limit controls
- 🔧 **Customizable**: Adjustable backing vocal volume (default 30%)
- 🚀 **GPU Acceleration**: Supports both CPU and GPU processing for faster separation
- 📝 **Ultrastar Integration**: Automatically adds `#VOCALS` and `#INSTRUMENTAL` tags to the UltraStar .txt files

## Installation

### Prerequisites

- Python 3.10 or higher
- FFmpeg installed and available in PATH

### GPU Version (Recommended)

For faster processing with NVIDIA GPU support:

```bash
# Clone the repository
git clone https://github.com/SecretUnicorn/UltrastarKaraokeMaker.git
cd UltrastarKaraokeMaker

# Create and activate virtual environment
python -m venv venv-cuda

venv-cuda\Scripts\activate  # Windows
# source venv-cuda/bin/activate  # Linux/Mac

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Install other dependencies
pip install -r cuda.requirements.txt
```

### CPU Version

For systems without compatible GPU:

```bash
# Clone the repository
git clone https://github.com/SecretUnicorn/UltrastarKaraokeMaker.git
cd UltrastarKaraokeMaker

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r cpu.requirements.txt
```

## Usage

### GPU Version (main_cuda.py)

Use this for faster processing with GPU acceleration:

```bash
python main_cuda.py "path/to/ultrastar/songs" [options]
```

**Options:**
- `--limit N`: Process only the first N folders
- `--offset N`: Skip the first N folders (useful for resuming)
- `--overwrite`: Process folders even if they already have vocals/instrumental files. If this flag is not set, folders with .txt files that contain either `#VOCALS` and `#INSTRUMENTAL` tags will be skipped.
- `--vocals_volume N`: Set backing vocals volume percentage (default: 30%)

**Examples:**
```bash
# Process all songs in a folder
python main_cuda.py "C:\Music\Ultrastar Songs"

# Process first 10 songs
python main_cuda.py "C:\Music\Ultrastar Songs" --limit 10

# Resume processing from song 51-100
python main_cuda.py "C:\Music\Ultrastar Songs" --limit 50 --offset 50

# Use 20% backing vocal volume and overwrite existing files
python main_cuda.py "C:\Music\Ultrastar Songs" --vocals_volume 20 --overwrite
```

### CPU Version (main.py)

Use this for systems without GPU support:

```bash
python main.py "path/to/ultrastar/songs" [options]
```

The options are the same as the GPU version, but processing will be slower.

## Acknowledgments

The separation logic implementation was inspired by [UltraSinger](https://github.com/rakuri255/UltraSinger?tab=MIT-1-ov-file#readme), which provides excellent tools for Ultrastar song creation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Troubleshooting

### Common Issues

1. **"FFmpeg not found"**: Make sure FFmpeg is installed and available in your system PATH
2. **CUDA errors**: Ensure you have compatible NVIDIA drivers and use the GPU installation method
3. **Out of memory**: Reduce batch processing or use CPU version for very long songs
4. **Model download fails**: Check your internet connection; models are downloaded automatically on first use

### Performance Tips

- Use the GPU version (`main_cuda.py`) for significantly faster processing
- Process songs in batches using `--limit` and `--offset` for better resource management
- Close other applications when processing large batches to free up system resources
