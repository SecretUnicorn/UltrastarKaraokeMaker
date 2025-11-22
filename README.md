# UltrastarKaraokeMaker

A powerful tool for automatically creating karaoke tracks from Ultrastar song folders. This tool separates vocals from instrumentals and creates karaoke-ready files with backing vocals at customizable volume levels.

## Features

- **Automatic Audio Separation**: Uses advanced AI models for high-quality vocal/instrumental separation
  - **GPU Mode**: `mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt` model via audio-separator
  - **CPU Mode**: Demucs HTDEMUCS model for CPU-friendly processing
- **Karaoke Track Generation**: Creates instrumental tracks with subtle backing vocals for better singing experience
- **Batch Processing**: Process multiple song folders at once with offset and limit controls
- **Customizable**: Adjustable backing vocal volume (default 40%)
- **UltraStar Integration**: Automatically adds `#VOCALS` and `#INSTRUMENTAL` tags to .txt files

## Installation

### Prerequisites

- Python 3.10 or higher
- Pip package manager
- FFmpeg installed and available in PATH
- Some patience and willingness to tinker because of potential dependency / hardware issues

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

### GPU Version Usage

Use this for faster processing with GPU acceleration:

```bash
python cuda.py "path/to/ultrastar/songs" --mode cuda [options]
```

**Options:**

- `--mode MODE`: Specify device mode (`cuda` or `cpu`); This setting is required.
- `--limit N`: Process only the first N folders
- `--offset N`: Skip the first N folders (useful for resuming)
- `--overwrite`: Process folders even if they already have vocals/instrumental files. If this flag is not set, folders with .txt files that contain either `#VOCALS` and `#INSTRUMENTAL` tags will be skipped.
- `--vocals_volume N`: Set backing vocals volume percentage (default: 40%)

**Examples:**

```bash
# Process all songs in a folder
python main.py "C:\Music\Ultrastar Songs"

# Process first 10 songs
python main.py "C:\Music\Ultrastar Songs" --limit 10

# Resume processing from song 51-100
python main.py "C:\Music\Ultrastar Songs" --limit 50 --offset 50

# Use 25% backing vocal volume and overwrite existing files
python main.py "C:\Music\Ultrastar Songs" --vocals_volume 25 --overwrite
```

### CPU Version Usage

Use this for systems without GPU support:

```bash
python main.py "path/to/ultrastar/songs" --mode cpu [options]
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
3. **audio_separator says gpu not available**: Reinstall the torch packages after you cleared the pip cache so that the correct ones get installed
4. **Model download fails**: Check your internet connection; models are downloaded automatically on first use

### Performance Tips

- Use the GPU version (`main.py --mode cuda`) for significantly faster processing
- Process songs in batches using `--limit` and `--offset` for better resource management
- Close other applications when processing large batches to free up system resources
