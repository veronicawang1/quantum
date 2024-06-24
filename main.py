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
    #return np.array([1*t, 2*t, 3*t, 4*t, 5*t, 6*t, 7*t, 8*t])

# FINDING LAMBDA
# Numerical differentiation using finite differences
def differentiate(time):
    return np.divide(np.subtract(ground_state(time + h), ground_state(time)), [h])


#abs value too and check size of h
#choose phase such that everything is real 
# normalize the eigevector, check to make sure numpy does this
# derivative is insensitive to h psi adds instead of subtract, leads to giant derivative, make a check against this

def norm(vector):
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        raise ValueError("Cannot norm a zero vector")
    return np.divide(vector, [magnitude])

def normalize(arr):
    norm_arr = []
    diff = 1
    diff_arr = np.max(arr) - np.min(arr)    
    for i in arr:
        temp = (((i - np.min(arr))*diff)/diff_arr)
        norm_arr.append(temp)
    return np.array(norm_arr)

'''
print("Original vector:", ground_state_derivative)
print("Vector norm:", norm_ground)
#should equal 1
print("Magnitude of norm:", np.linalg.norm(norm_ground))

# Print the results
print("Ground state eigenvector at time =", time_value, ":\n", ground_state_at_time)
print("Derivative of the ground state eigenvector at time =", time_value, ":\n", ground_state_derivative)
'''
def integralof(values, A, B):
    # Generate an array of x values from A to B
    # x = np.linspace(A, B)
    # Compute the trapezoidal sum using numpy's trapz function
    return np.trapz(values)

def integrate_derivative(start_time, end_time):
    # t_values = np.linspace(start_time, end_time, h2)
    t_values = np.linspace(start_time, end_time, h2)

    derivatives = np.array([differentiate(t) for t in t_values])
    #print(f"derivatives: {derivatives}")
    dt = (end_time - start_time) / (h2 - 1)
    # Initialize an empty list to hold the integrals
    integral = []
    
    # Iterate over each column
    for j in range(8):
        # Extract the j-th column
        column = derivatives[:, j]
        #print(f"column: {column}")
        # Compute the integral for the j-th column using the specified A and B
        integral.append(np.sum(column, 0, dtype = np.float32) * dt)
        #print(f"integral: {integral}")
    
    return integral

path_length = 0
def integrate_norm(start_time, end_time):
    path_length = 0
    t_values = np.linspace(start_time, end_time, h2)
    dt = (end_time - start_time) / (h2 - 1)

    derivatives = np.array([norm(differentiate(t)) for t in t_values])
    # print(derivatives)

    # Integration without norming, a check

    integral = np.zeros(derivatives.shape[1])
    for i in range(h2):
        path_length += (derivatives[i]) * dt  
    return path_length

path_length = integrate_norm(1, 10)

# Perform the integration

integrated_result = integrate_derivative(initial_time, end_time)

# Print the results
print("Integrated result from time", initial_time, "to", end_time, ":\n", integrated_result)
true_result = np.subtract(ground_state(end_time), ground_state(initial_time))

print(f"The true result should be \n{true_result}")

arr = [1, 2, 3]
print(normalize(arr))

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
# plt.plot(x1, y2)
# plt.show()
