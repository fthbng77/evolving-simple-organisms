from random import uniform


class Food():
    def __init__(self, settings):
        self.respawn(settings)

    def respawn(self, settings):
        self.x = uniform(settings['x_min'], settings['x_max'])
        self.y = uniform(settings['y_min'], settings['y_max'])
        self.energy = 1