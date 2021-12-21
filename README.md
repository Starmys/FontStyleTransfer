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
python main.py --render             # Render character images
python main.py --train              # Train and/or test a model
```

## Configuration
Modify ```config.yaml``` before training.
```yaml
general:
  size: 32                 # Image size
  encoding: 'ASCII'        # Character encoding
  characters:              # Character ranges
    - from: 'a'
      to: 'z'
    - from: 'A'
      to: 'Z'
    - from: '0'
      to: '9'
render:
  from: 'data/raw'         # Source directorry
  to: 'data/img'           # Destination directorry
  fonts: ['*']             # Render all fonts in the source directorry
train:
  name: 'ex0'              # Experiment tag
  data:
    dir: 'data/img'        # Should be the same as config['render']['to']
    basefont: 'comic'
    trainfonts: ['bradhitc', 'caveat', 'fuzzybubbles', 'indieflower', 'inkfree', 'mali', 'ole', 'sacramento', 'shadowsintolight', 'twinklestar']
    testfonts: ['cookie', 'gloria', 'vujahdayscript']
    seed: 2021             # Random seed for data dividing
  generator: 'u_net'       # 'u_net' or 'style_generator'
  discriminator: 'classifier'  # No other options
  training:
    seed: 2021             # Random seed in training
    lr: 0.01               # Learning rate
    epoch: 100             # Epoch number
    iteration: 2000        # Iteration number in each epoch
    batch_size: 5          # Batch size
    g_d_rate: 10           # Generator / Discriminator training frequency ratio
    loss:
      gan_loss: 'lsgan'    # 'lsgan' or 'wgan'
      l1_loss_weight: 10
      l2_loss_weight: 0
```