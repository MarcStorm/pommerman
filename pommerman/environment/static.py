from pommerman.envs.v0 import Pomme
import os
import json

PATH = os.path.dirname(os.path.realpath(__file__))

STATE_FILE = os.path.join(PATH, './pommerman.board.json')

class StaticPomme(Pomme):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_init_game_state(STATE_FILE)
        self.is_locked = False

    
    def reset(self):
        if not self.is_locked:
            self.set_init_game_state(STATE_FILE)
            self.is_locked = True
        return super().reset()


    # def lock_game_state(self):
    #     with open(STATE_FILE, 'w') as f:
    #         info = self.get_json_info()
    #         f.write(json.dumps(info, sort_keys=True, indent=4))
    #     self.set_init_game_state(STATE_FILE)
    #     self.is_locked = True

    