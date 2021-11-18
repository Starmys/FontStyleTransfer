class Characters(object):

    def __init__(self, config):
        try:
            self.size = config['size']
            if config['encoding'].upper() != 'ASCII':
                raise AssertionError(f'{config["encoding"]} encoding is not supported.')
            self.characters = []
            for char_set in config['characters']:
                self.characters += [chr(i) for i in range(ord(char_set['from']), ord(char_set['to']) + 1)]
        except AssertionError as err:
            exit(err)
        except:
            exit('General config error.')

    def __str__(self):
        return str(self.characters)
