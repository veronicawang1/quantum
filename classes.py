import random
import math
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from coupling import *

class Sine:
    def __init__(self):
        self.amp = random.gauss(0, 1)
        self.phase = random.gauss(0, 2*math.pi)
        #??
        self.period = random.gauss(0, 2*math.pi)
    def value(self, time):
        #print(f"{self.amp} * sin({self.period}({time}-{self.phase}))\n")
        return self.amp*math.sin(self.period*(time-self.phase))

class JValue:
    def __init__(self, n):
        self.numSineCurves = n
        self.sines = []
        for i in range(n):
            self.sines.append(Sine())
    def showValue(self, t):
        j = 0
        for i in range(self.numSineCurves):
            j += self.sines[i].value(t)
        #print(j)
        return j
    