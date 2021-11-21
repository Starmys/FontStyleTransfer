import os

import torch
import numpy as np
from PIL import Image


class Data(object):

    def __init__(self, config):

        self._basefont = {}
        basefont_dir = os.path.join(config['dir'], config['basefont'])
        for file in os.listdir(basefont_dir):
            char = os.path.splitext(file)[0]
            path = os.path.join(basefont_dir, file)
            self._basefont[char] = path

        if 'seed' in config:
            np.random.seed(config['seed'])

        self._paths = {'train': {}, 'dev': {}, 'test': {}}
        for font in os.listdir(config['dir']):
            font_dir = os.path.join(config['dir'], font)
            if font != config['basefont']:
                self._paths['train'][font] = {}
                self._paths['dev'][font] = {}
                self._paths['test'][font] = {}
                for file in os.listdir(font_dir):
                    char = os.path.splitext(file)[0]
                    if char not in self._basefont:
                        continue
                    path = os.path.join(font_dir, file)
                    p = np.random.random()
                    if p < config['train']:
                        self._paths['train'][font][char] = path
                    elif p < config['train'] + config['dev']:
                        self._paths['dev'][font][char] = path
                    else:
                        self._paths['test'][font][char] = path

        self._cache = {}

    def _to_torch(self, x):
        return torch.reshape(torch.from_numpy(x), (1, 1, x.shape[0], x.shape[1]))

    def _load_image(self, path):
        if path in self._cache:
            return self._cache[path]
        else:
            with Image.open(path) as img:
                img_data = np.asarray(img)[:, :, 0]
            img_data = 1 - img_data.astype(np.float32) / 255 * 2
            img_data = self._to_torch(img_data)
            self._cache[path] = img_data
            return img_data

    def _wrap_data(self, mode, font, char_x, char_y):
        x1 = self._load_image(self._basefont[char_x])
        x2 = self._load_image(self._paths[mode][font][char_y])
        y = self._load_image(self._paths[mode][font][char_x])
        return f'{font}: {char_x} => {char_y}', x1, x2, y

    def iterator(self, mode):
        fonts = list(self._paths[mode].keys())
        if mode == 'train':
            while True:
                font = fonts[np.random.choice(len(fonts))]
                chars = list(self._paths[mode][font].keys())
                np.random.shuffle(chars)
                char_x, char_y = chars[:2]
                yield self._wrap_data(mode, font, char_x, char_y)
        else:
            for font in self._paths[mode]:
                chars = self._paths[mode][font].keys()
                for char_x in chars:
                    for char_y in chars:
                        if char_x != char_y:
                            yield self._wrap_data(mode, font, char_x, char_y)

    def plot_results(self, name, tag, msg, y, y_hat):
        font, chars = msg.split(': ')
        char_1, char_2 = chars.split(' => ')
        font_dir = os.path.join('logs', name, tag, font)
        if not os.path.exists(font_dir):
            os.makedirs(font_dir)
        img_data = torch.cat((y, y_hat), -1)
        img_data = img_data.detach().numpy().reshape(img_data.shape[-2:])
        img_data = ((1 - img_data) / 2 * 255).astype(np.uint8)
        img_data = np.repeat(img_data[:, :, np.newaxis], 3, axis=-1)
        Image.fromarray(img_data).save(os.path.join(font_dir, f'{char_1}.{char_2}.jpg'))
