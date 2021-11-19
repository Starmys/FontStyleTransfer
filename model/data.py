import os

import numpy as np
from matplotlib import image


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

    def _load_image(self, path):
        if path in self._cache:
            return self._cache[path]
        else:
            img = image.imread(path)[:, :, 0]
            img = 1 - img.astype(np.float32) / 255 * 2
            self._cache[path] = img
            return img

    def _wrap_data(self, mode, font, char_x, char_y):
        x1 = self._load_image(self._basefont[char_x])
        x2 = self._load_image(self._paths[mode][font][char_x])
        y = self._load_image(self._paths[mode][font][char_y])
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
