from typing import Iterable


class Characters(Iterable):

    def __init__(self, config):
        try:
            self._size = config['size']
            self._characters = []
            if config['encoding'].upper() == 'ASCII':
                for char_set in config['characters']:
                    self._characters += [chr(i) for i in range(ord(char_set['from']), ord(char_set['to']) + 1)]
            else:
                raise AssertionError(f'{config["encoding"]} encoding is not supported.')
        except AssertionError as err:
            exit(err)
        except:
            exit('General config error.')

    def __str__(self):
        return str(self._characters)

    def __iter__(self):
        return iter(self._characters)

    def get_size(self):
        return self._size
