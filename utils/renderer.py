import os
import glob

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .characters import Characters


class Renderer(object):

    def __init__(self, config, chars: Characters):
        self._chars = chars
        try:
            font_paths = set()
            for condition in config['fonts']:
                condition = condition if condition[-4:] == '.ttf' else f'{condition}.ttf'
                font_paths |= set(glob.glob(os.path.join(config['from'], condition)))
            self._tasks = []
            for font_path in font_paths:
                font_name = os.path.splitext(os.path.split(font_path)[1])[0]
                self._tasks.append({
                    'from': font_path,
                    'to': os.path.join(config['to'], font_name)
                })
        except:
            exit('Render config error.')

    def start(self):
        size = self._chars.size
        for task in self._tasks:
            print(f'Rendering font: {task["from"]} => {os.path.join(task["to"], "*.jpg")}')
            if not os.path.exists(task['to']):
                os.makedirs(task['to'])
            font = ImageFont.truetype(task['from'], int(size * 0.9))
            for char in self._chars.characters:
                image = Image.fromarray(np.zeros((size, size, 3)).astype(np.uint8) - 1)
                draw = ImageDraw.Draw(image)
                w, h = font.getsize(char)
                draw.text(((size - w) / 2, (size * 0.9 - h) / 2), char, 'black', font, align='center')
                filename = f'{char.lower()}+' if char.isupper() else char
                image.save(os.path.join(task['to'], f'{filename}.jpg'))
        print('Rendering accomplished.')
