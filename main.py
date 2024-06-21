import random
import math
import numpy as np
import sympy
import matplotlib.pyplot as plt
from qutip import *
from sympy import *
from coupling import *
from classes import *
from hamiltonian import *
from constants import *


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
def differentiate(vector, time):
    return (vector(time + h) - vector(time)) / (h)


#abs value too and check size of h

def normalize(vector):
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        raise ValueError("Cannot normalize a zero vector")
    return vector / magnitude


'''
print("Original vector:", ground_state_derivative)
print("Normalized vector:", normalized_ground)
#should equal 1
print("Magnitude of normalized vector:", np.linalg.norm(normalized_ground))

# Print the results
print("Ground state eigenvector at time =", time_value, ":\n", ground_state_at_time)
print("Derivative of the ground state eigenvector at time =", time_value, ":\n", ground_state_derivative)
'''

# Integrate the derivative using the trapezoidal rule
def integrate_derivative(start_time, end_time):
    t_values = np.linspace(start_time, end_time, int(1/h))
    dt = (end_time - start_time) / (int(1/h) - 1)

    derivatives = np.array([differentiate(ground_state, t) for t in t_values])

    for t in t_values:
        print(ground_state(t))

    # Integration without norming, a check
    """
    integral = np.zeros(derivatives.shape[1])
    for i in range(1, int(1/h)):
        integral += (derivatives[i]) * dt  
    return integral
    """

    integral = 0
    for i in range(1, int(1/h)):
        integral += (normalize(derivatives[i])) * dt  
    return integral


# Perform the integration
integrated_result = integrate_derivative(initial_time, end_time)

# Print the results
print("Integrated result from time", initial_time, "to", end_time, ":\n", integrated_result)
true_result = np.subtract(ground_state(end_time), ground_state(initial_time))

print(f"The true result should be \n{true_result}")
##############################
#PLOTTING
a = JValue(numberOfSine)
#view_arr(hamMatrix)
#view_arr(matrix)
#higher third argument = smoother?
x = np.linspace(0, 2*math.pi, 200)
y = ([a.showValue(x_value) for x_value in x])

x1 = np.linspace(initial_time, end_time, 100)
y1 = ([ground_state(t) for t in x1])

y2 = ([integrate_derivative(0, t) for t in x1])
# plt.plot(x, y)
#plt.plot(x1, y1)
plt.plot(x1, y2)
plt.show()
