import random
import math
import numpy as np
import sympy
import matplotlib.pyplot as plt
from qutip import *
from sympy import *
from coupling import *
from classes import *
from constants import *

rows = ["111", "110", "101", "100", "011", "010", "001", "000"]
columns = ["111", "110", "101", "100", "011", "010", "001", "000"]
matrix = [["" for _ in range(8)] for _ in range(8)]

# Sine curves used are constant each time you run code
curve1 = JValue(numberOfSine)
curve2 = JValue(numberOfSine)
curve3 = JValue(numberOfSine)
j12 = curve1.showValue(initial_time)
j13 = curve2.showValue(initial_time)
j23 = curve3.showValue(initial_time)

def calculate_ham_matrix(t):
    matrix = []
    j12 = curve1.showValue(t)
    j13 = curve2.showValue(t)
    j23 = curve3.showValue(t)
    for i in range(8):
        tempArr = []
        state1 = rows[i]
        for j in range(8):
            tempVal = 0
            state2 = columns[j]
            # change number of sig figs?
            tempVal = round(
                    ((j12 * ((-1) ** compareState(state1[0], state2[1]))) + 
                    (j13 * ((-1) ** compareState(state1[0], state2[2]))) + 
                    (j23 * ((-1) ** compareState(state1[1], state2[2])))), 3)
            tempArr.append(tempVal)
            #print(f"{j12} * (-1 ** {compareState(state1[0], state2[1])})")
            #print((-1) ** compareState(state1[0], state2[1]))
        matrix.append(tempArr)
    return matrix


def view_arr(m):
    for i in range(8):
        print(str(m[i]))