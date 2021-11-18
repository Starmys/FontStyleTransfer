# Font Style Transfer

## Requirements
- Python 3.6
- PyTorch 1.8.1

## Install
```bash
pip install -r requirements.txt
```

## Quickstart
```bash
cp config.example.yaml config.yaml  # Create a config file
python main.py --render  # Render character images
python main.py --train  # Train and/or test a model
python main.py --generate  # Generate a new font
```

## Configuration