import numpy as np
import matplotlib.animation as animate

class Affiche_NCA():
    def __init__(self, input, color_map):
        self.input = input
        self.color_map = color_map
        self.transform()
        
    def transform(self):
        return self.input
    
    def next(self):
        n,p = self.input.shape
        return np.random.rand(n,p,3)
