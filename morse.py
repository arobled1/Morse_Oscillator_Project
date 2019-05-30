#=============================================================================80
# Description:
# Computes an eigenvalue decomposition of the hamiltonian for a 1D morse
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
import matplotlib.pyplot as plt

def generate_grid(xmin, xmax, ngrid):
    dx = (xmax - xmin) / (ngrid - 1)
    x_grid = [xmin + i * dx for i in xrange(ngrid)]
    return dx, x_grid

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

def get_potential(ngrid):
    pe_matrix = np.zeros((ngrid, ngrid))
    for i in xrange(ngrid):
        pe_matrix[i][i] = d_well * (np.exp(-omegax * x_grid[i]) - 1)**2
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
xmin = -4.0
xmax = 12.0
ngrid = 300

dx, x_grid = generate_grid(xmin, xmax, ngrid)
ke_matrix = get_kinetic(ngrid, dx)
pe_matrix = get_potential(ngrid)
hamiltonian = ke_matrix + pe_matrix
eig_val, eig_vec = la.eig(hamiltonian)
sort_eigval, sort_eigvec = bubble_sort(eig_val, eig_vec)
ground = [sort_eigvec[i][0] - 0.3 for i in xrange(ngrid)]
first_exc = [sort_eigvec[i][1] - 0.1 for i in xrange(ngrid)]
sec_exc = [sort_eigvec[i][2] + 0.1 for i in xrange(ngrid)]
third_exc = [sort_eigvec[i][3] + 0.3 for i in xrange(ngrid)]
fourth_exc = [sort_eigvec[i][4] + 0.5 for i in xrange(ngrid)]
potential = [d_well * (np.exp(-omegax * x_grid[i]) - 1)**2 for i in xrange(ngrid)]
plt.xlim(min(x_grid) - 0.1, max(x_grid) + 0.1)
plt.ylim(min(ground) - 0.1, max(fourth_exc) + 0.1)
plt.yticks([])
plt.plot(x_grid, ground, label='n = 0')
plt.plot(x_grid, first_exc, label='n = 1')
plt.plot(x_grid, sec_exc, label='n = 2')
plt.plot(x_grid, third_exc, label='n = 3')
plt.plot(x_grid, fourth_exc, label='n = 4')
plt.xlabel("x")
plt.ylabel("psi")
plt.legend()
plt.savefig("wavefunctions.pdf")
plt.clf()
f = open('eigenvalues.dat', 'wb')
f.write("n       E_n\n")
for i in xrange(ngrid):
    f.write("%s %s\n" % (i, sort_eigval[i]))
f.close()
