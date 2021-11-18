import os
import yaml
import argparse

from utils.characters import Characters


if __name__ == '__main__':

    if not os.path.exists('config.yaml'):
        exit('Config file not found.')
    
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    chars = Characters(config['general'])
    print(chars)

    parser = argparse.ArgumentParser()
    parser.add_argument('--render', '-r', help='Render character images', default=False, action='store_true')
    parser.add_argument('--train', '-t', help='Train and/or test a model', default=False, action='store_true')
    parser.add_argument('--generate', '-g', help='Generate a new font', default=False, action='store_true')
    args = parser.parse_args()

    if args.render:
        exit()
