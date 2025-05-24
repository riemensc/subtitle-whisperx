# Karaoke Maker 🎤

A powerful Python tool that creates karaoke-style videos with word-by-word highlighting effects using AI-powered speech recognition. Transform any video into an engaging karaoke experience with synchronized subtitles that highlight each word as it's spoken.

## ✨ Features

- **AI-Powered Transcription**: Uses Faster-Whisper for accurate speech-to-text conversion
- **Word-Level Synchronization**: Creates precise timing for each word
- **Multiple Visual Styles**: 7 different subtitle styles to choose from
- **Karaoke Effects**: Real-time word highlighting with smooth transitions
- **Custom Text Support**: Use your own text instead of auto-transcription
- **GPU Acceleration**: CUDA support for faster processing
- **Flexible Output**: Generates both SRT subtitle files and final video

## 🎨 Available Styles

1. **social_fixed_yellow** - Bold yellow highlighting, perfect for social media
2. **social_fixed_green** - Vibrant green highlight effect
3. **subtle_highlight_orange** - Elegant orange highlighting with larger font
4. **clean_blue** - Clean blue style with Calibri font
5. **minimal_cyan** - Minimalist cyan highlighting
6. **warm_red_highlight** - Warm red highlighting with shadow effects
7. **simple_bold_yellow_small** - Compact bold yellow style

## 📋 Requirements

### System Requirements
- Python 3.7 or higher
- FFmpeg (for video processing)
- CUDA-capable GPU (optional, for faster processing)

### Python Dependencies
- faster-whisper
- pysrt
- torch (for GPU acceleration)

## 🚀 Installation

### Windows

1. **Install Python**
   ```bash
   # Download from https://www.python.org/downloads/
   # Make sure to check "Add Python to PATH" during installation
   ```

2. **Install FFmpeg**
   ```bash
   # Option 1: Using Chocolatey
   choco install ffmpeg
   
   # Option 2: Download from https://ffmpeg.org/download.html
   # Extract and add to system PATH
   ```

3. **Install Python dependencies**
   ```bash
   pip install faster-whisper pysrt torch
   
   # For CUDA support (if you have NVIDIA GPU)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### macOS

1. **Install Python**
   ```bash
   # Using Homebrew
   brew install python
   
   # Or download from https://www.python.org/downloads/
   ```

2. **Install FFmpeg**
   ```bash
   brew install ffmpeg
   ```

3. **Install Python dependencies**
   ```bash
   pip3 install faster-whisper pysrt torch
   ```

### Linux (Ubuntu/Debian)

1. **Install Python and FFmpeg**
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip ffmpeg
   ```

2. **Install Python dependencies**
   ```bash
   pip3 install faster-whisper pysrt torch
   
   # For CUDA support (if you have NVIDIA GPU)
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Linux (CentOS/RHEL/Fedora)

1. **Install Python and FFmpeg**
   ```bash
   # CentOS/RHEL
   sudo yum install python3 python3-pip ffmpeg
   
   # Fedora
   sudo dnf install python3 python3-pip ffmpeg
   ```

2. **Install Python dependencies**
   ```bash
   pip3 install faster-whisper pysrt torch
   ```

## 🔧 Usage

### Basic Usage

```bash
python subtile-whisperx.py --video input_video.mp4
```

### Advanced Usage

```bash
# Specify output file and style
python subtile-whisperx.py --video input.mp4 --output karaoke_output.mp4 --style social_fixed_green

# Use custom text instead of transcription
python subtile-whisperx.py --video input.mp4 --text "Your custom lyrics here" --style clean_blue

# Specify language and model
python subtile-whisperx.py --video input.mp4 --language de --model large --style minimal_cyan

# Customize words per line
python subtile-whisperx.py --video input.mp4 --words-per-line 5 --style warm_red_highlight
```

### Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--video` | `-v` | Input video file (required) | - |
| `--output` | `-o` | Output video file | Auto-generated |
| `--style` | `-s` | Subtitle style | social_fixed_yellow |
| `--text` | `-t` | Custom text instead of transcription | None |
| `--model` | `-m` | Whisper model size | medium |
| `--language` | `-l` | Language code (e.g., 'de', 'en') | Auto-detect |
| `--min-segment-length` | - | Minimum word duration in seconds | 0.1 |
| `--words-per-line` | - | Words per karaoke line | 7 |
| `--debug` | `-d` | Enable debug output | False |
| `--log` | - | Log file path | karaoke_maker.log |
| `--check` | `-c` | Check installation | False |

### Whisper Model Sizes

- `tiny` - Fastest, least accurate
- `base` - Good balance of speed and accuracy
- `small` - Better accuracy, slower
- `medium` - **Default**, good accuracy
- `large` - Best accuracy, slowest

## 🛠️ Installation Check

Verify your installation:

```bash
python subtile-whisperx.py --check
```

This will check:
- FFmpeg installation and version
- Faster-Whisper availability
- CUDA support (if available)
- All dependencies

## 📁 Output Files

The script generates:

1. **Karaoke Video** - Main output with synchronized subtitles
2. **SRT File** - Subtitle file with word-level timing
3. **Log File** - Detailed processing information

Example output:
```
karaoke_input_video_social_fixed_yellow_20250524_143022.mp4
karaoke_input_video_social_fixed_yellow_20250524_143022.final.srt
karaoke_maker.log
```

## 🎯 Examples

### Create a TikTok-style karaoke video
```bash
python subtile-whisperx.py --video dance_video.mp4 --style social_fixed_yellow --words-per-line 5
```

### Process a German video with custom styling
```bash
python subtile-whisperx.py --video german_song.mp4 --language de --style clean_blue --model large
```

### Use custom lyrics
```bash
python subtile-whisperx.py --video music_video.mp4 --text "Never gonna give you up, never gonna let you down" --style warm_red_highlight
```

## 🐛 Troubleshooting

### Common Issues

1. **FFmpeg not found**
   ```
   Error: FFmpeg not found
   Solution: Install FFmpeg and add to system PATH
   ```

2. **CUDA out of memory**
   ```
   Solution: Use CPU instead or smaller model
   python subtile-whisperx.py --video input.mp4 --model small
   ```

3. **Audio extraction fails**
   ```
   Solution: Check video file format and codec support
   ```

4. **Empty transcription**
   ```
   Solution: Check audio quality or use custom text
   python subtile-whisperx.py --video input.mp4 --text "fallback text"
   ```

### Performance Tips

- Use GPU acceleration when available
- Choose smaller Whisper models for faster processing
- Reduce video resolution for quicker encoding
- Use SSD storage for better I/O performance

## 📝 License

This project is open source. Please check the license file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 🔍 Technical Details

- **Audio Processing**: Extracts 16kHz mono WAV audio from input video
- **Speech Recognition**: Uses Faster-Whisper with beam search and VAD filtering
- **Subtitle Format**: Generates ASS format with karaoke timing effects
- **Video Encoding**: Uses FFmpeg with H.264 codec and AAC audio

## 📞 Support

If you encounter issues:

1. Check the log file for detailed error information
2. Run with `--debug` flag for verbose output
3. Use `--check` to verify installation
4. Open an issue on GitHub with log details
