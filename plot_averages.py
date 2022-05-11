#from sys import last_value
import numpy as np
import matplotlib.pyplot as plt

class plot_averages():
    def __init__(self, moving_episodes = 100):
        self.moving_episodes = 100
        self.last_values = []
        self.last_averages = []

    
    def save_averages(self):
        self.last_averages.append(np.mean(self.last_values))

    def add_value(self,value):
        if len(self.last_values) == 100:
            self.last_values.pop()
            self.last_values.insert(0,value)
        else: 
            self.last_values.append(value)
        self.save_averages()

    def plot_averages(self):
        print(self.last_averages)
        plt.plot(range(1,len(self.last_averages)+1),self.last_averages[:])
        plt.show()
        plt.draw()
        plt.close()
    