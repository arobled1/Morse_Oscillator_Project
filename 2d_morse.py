#=============================================================================80
# Description:
# Computes an eigenvalue decomposition of the hamiltonian for a 2D morse
# oscillator.
#===============================================================================
# Author: Alan Robledo
# Date modified: May 30, 2019
#===============================================================================
# Variables:
# xbound = sets the bound in the positive and negative direction
# ngrid = number of grid points
# dx = dist. between points (Remember: for every ngrid points, there
#       are ngrid - 1 dx)
# d_well = depth of the morse potential well
# mass = reduced mass
#===============================================================================
import numpy as np
from scipy import linalg as la
import copy
#import matplotlib.pyplot as plt

def generate_grid(min, max, ngrid):
    dx = (max - min) / (ngrid - 1)
    grid = [min + i * dx for i in xrange(ngrid)]
    return dx, grid

def get_kinetic(ngrid, dx):
    ke_matrix = np.zeros((ngrid, ngrid))
    ke_matrix[0][0] = -2
    ke_matrix[1][0] = 1
    ke_matrix[ngrid - 1][ngrid - 1] = -2
    ke_matrix[ngrid - 2][ngrid - 1] = 1
    for i in xrange(1, ngrid - 1):
        ke_matrix[i - 1][i] = 1
        ke_matrix[i][i] = -2
        ke_matrix[i + 1][i] = 1
    return -(hbar**2 / (2.0 * mass)) * dx**-2 * ke_matrix

def get_potential(ngrid, d_well, omega):
    pe_matrix = np.zeros((ngrid, ngrid))
    for i in xrange(ngrid):
        pe_matrix[i][i] = d_well * (np.exp(-omega * grid[i]) - 1)**2
    return pe_matrix

def bubble_sort(eig_energies, eig_vectors):
    new_list = copy.copy(eig_energies)
    new_mat = copy.deepcopy(eig_vectors)
    num_pairs = len(new_list) - 1
    for j in xrange(num_pairs):
        for i in xrange(num_pairs - j):
            if new_list[i] > new_list[i+1]:
                new_list[i], new_list[i+1] = new_list[i+1], new_list[i]
                new_mat[:,[i, i+1]] = new_mat[:,[i+1,i]]
    return new_list, new_mat

d_well = 12
hbar = 1
mass = 1
omegax = 0.2041241
omegay = 0.18371169
grid_min = -4.0
grid_max = 12.0
ngrid = 300

# Defining the grid
dx, grid = generate_grid(grid_min, grid_max, ngrid)
# Getting kinetic energy matrix
ke_matrix = get_kinetic(ngrid, dx)
# Kinetic energy in x is the same as in y so we can just multiply by 2.
ke_matrix = 2 * ke_matrix
# Getting potential energy matrix
pex_matrix = get_potential(ngrid, d_well, omegax)
pey_matrix = get_potential(ngrid, d_well, omegay)
# Making hamiltonian matrix
hamiltonian = ke_matrix + pex_matrix + pey_matrix
# Eigenvalue decomposition of hamiltonian
eig_val, eig_vec = la.eig(hamiltonian)
# Sorting the eigenvalues and corresponding eigenvectors
sort_eigval, sort_eigvec = bubble_sort(eig_val, eig_vec)
f = open('2d_eigenvalues.dat', 'wb')
f.write("n       E_n\n")
for i in xrange(ngrid):
    f.write("%s %s\n" % (i, sort_eigval[i]))
f.close()
