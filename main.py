import random
import math
import numpy as np
import sympy
import matplotlib.pyplot as plt
from qutip import *
from sympy import *
from coupling import *
from classes import *
initial_time = 0
end_time = 10
numberOfSine = 5

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

hamMatrix = []
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


for i in range(8):
    for j in range(8):
        matrix[i][j] = coupleState(rows[i], columns[j])

###########################
### REMEMBER TO CONVERT MATRIX INTO NUMPY MATRIX EVERY TIME YOU CHANGE IT
#EIGENVALUES, EIGENSTATES, DIAGONALIZATION
hamMatrix = calculate_ham_matrix(initial_time)
hamMatrixNp = np.array(hamMatrix)


# Compute ground state eigenvector
def ground_state(t):
    eigenvalues, eigenvectors = np.linalg.eigh(calculate_ham_matrix(t))
    # Form the diagonal matrix from the eigenvalues
    # diagonalMatrix = np.diag(eigenvalues)
    # The matrix of eigenvectors
    # P = eigenvectors
    # Verify the diagonalization: A should be equal to P * D * P_inv
    # P_inv = np.linalg.inv(P)
    # A_diag = P @ diagonalMatrix @ P_inv
    min_index = np.argmin(eigenvalues)
    return eigenvectors[:, min_index]

# FINDING LAMBDA
# Numerical differentiation using finite differences
def differentiate(vector, time, h=1e-5):
    return (vector(time + h) - vector(time - h)) / (2 * h)

# Compute the numerical derivative of the ground state eigenvector
time_value = 1.0
ground_state_at_time = ground_state(time_value)
ground_state_derivative = differentiate(ground_state, time_value)

# Print the results
print("Ground state eigenvector at time =", time_value, ":\n", ground_state_at_time)
print("Derivative of the ground state eigenvector at time =", time_value, ":\n", ground_state_derivative)

# Integrate the derivative using the trapezoidal rule
def integrate_derivative(A, B, num_points=1000):
    t_values = np.linspace(A, B, num_points)
    dt = (B - A) / (num_points - 1)

    derivatives = np.array([differentiate(ground_state, t) for t in t_values])
    
    # Trapezoidal integration
    integral = np.zeros(derivatives.shape[1])
    for i in range(1, num_points):
        integral += (derivatives[i] + derivatives[i-1]) * dt / 2   
    return integral


# Perform the integration
integrated_result = integrate_derivative(initial_time, end_time)

# Print the results
print("Integrated result from time", initial_time, "to", end_time, ":\n", integrated_result)

##############################
#PLOTTING
a = JValue(numberOfSine)
#view_arr(hamMatrix)
#view_arr(matrix)
#higher third argument = smoother?
x = np.linspace(0, 2*math.pi, 200)
y = ([a.showValue(x_value) for x_value in x])

#plt.plot(x, y)
#plt.show()
