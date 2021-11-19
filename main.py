import os
import yaml
import argparse

from utils.characters import Characters
from utils.renderer import Renderer
from model.GAN import GAN


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--render', '-r', help='Render character images', default=False, action='store_true')
    parser.add_argument('--train', '-t', help='Train and/or test a model', default=False, action='store_true')
    parser.add_argument('--generate', '-g', help='Generate a new font', default=False, action='store_true')
    args = parser.parse_args()

    if not os.path.exists('config.yaml'):
        exit('Config file not found.')

    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    for section in ['general'] + [k for k, v in vars(args).items() if v]:
        if section not in config.keys():
            exit(f'Missing {section} config.')

    chars = Characters(config['general'])

    if args.render:
        Renderer(config['render'], chars).start()
    elif args.train:
        GAN(config['train']).start()
    elif args.generate:
        exit('Not implemented.')
    else:
        parser.print_help()
