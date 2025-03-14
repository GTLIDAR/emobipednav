import pysocialforce as psf

class SFM:
    def __init__(self):
        self.name = 'SFM'
        self.config = None
        
    def set_config(self, config):
        self.config = config
        
    def predict(self, state):
        pedestrians = state['pedestrians']
        obstacles = state['obstacles']
        
        s = psf.Simulator(
            pedestrians,
            self.config,
            obstacles=obstacles
        )
        s.step_once()
        
        ped_states, _ = s.get_states()
        
        # the last ped is the robot
        return ped_states[-1][:-1]
   