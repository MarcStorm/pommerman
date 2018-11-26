from pommerman.envs.v0 import Pomme

import json

STATE_FILE = './pommerman.json'

class StaticPomme(Pomme):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_locked = False

    
    def reset(self):
        obs = super().reset()
        if not self.is_locked:
            self.lock_game_state()
        return obs


    def lock_game_state(self):
        with open(STATE_FILE, 'w') as f:
            info = self.get_json_info()
            f.write(json.dumps(info, sort_keys=True, indent=4))
        self.set_init_game_state(STATE_FILE)
        self.is_locked = True

    