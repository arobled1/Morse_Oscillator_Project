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

def get_transition(eig_vec, ngrid, state1, state2):
    tmp = np.zeros((ngrid, ngrid))
    for i in xrange(ngrid):
        for j in  xrange(ngrid):
            tmp[j,i] = eig_vec[j,state1] * eig_vec[i,state2]
    return tmp

d_well = 12
hbar = 1
mass = 1
omegax = 0.2041241
xmin = -6.0
xmax = 12.0
ngrid = 300

# Defining the grid
dx, x_grid = generate_grid(xmin, xmax, ngrid)

# Getting kinetic energy matrix
ke_matrix = get_kinetic(ngrid, dx)

# Getting potential energy matrix
pe_matrix = get_potential(ngrid)

# Making hamiltonian matrix
hamiltonian = ke_matrix + pe_matrix

# Eigenvalue decomposition of hamiltonian
eig_val, eig_vec = la.eig(hamiltonian)

# Sorting the eigenvalues and corresponding eigenvectors
sort_eigval, sort_eigvec = bubble_sort(eig_val, eig_vec)
if np.sign(sort_eigvec[0][0]) == -1:
    sort_eigvec = sort_eigvec * -1

# Creating the transition matrix for going from ground to 1st excited state
ground_first = get_transition(sort_eigvec, ngrid, 0, 1)
ground_second = get_transition(sort_eigvec, ngrid, 0, 2)
u, s, vt = la.svd(ground_first)
if np.sign(u[0][0]) == -1:
    u = -1 * u
    vt = -1 * vt

plot_u = np.array([u[i][0] for i in xrange(ngrid)])
plot_vt = np.array([vt[0][i] for i in xrange(ngrid)])
plt.xlim(min(x_grid) - 0.1, max(x_grid) + 0.1)
plt.ylim(min(plot_u) - 0.1, max(plot_vt) + 0.4)
plt.yticks([])
plt.plot(x_grid, plot_u, label=r'$\psi_{hole}$', color='blue')
plt.plot(x_grid, plot_vt + 0.3, label=r'$\psi_{electron}$', color='black')
plt.axhline(y = 0, linestyle='dashed', color='grey')
plt.axhline(y = 0.3, linestyle='dashed', color='grey')
plt.legend()
plt.savefig("morse_ntos.pdf")
plt.clf()

# Next 6 lines are solely for making plots of wavefunctions
ground = np.array([sort_eigvec[i][0] for i in xrange(ngrid)])
first_state = np.array([sort_eigvec[i][1] for i in xrange(ngrid)])
sec_state = np.array([sort_eigvec[i][2] for i in xrange(ngrid)])
third_state = np.array([sort_eigvec[i][3] for i in xrange(ngrid)])
fourth_state = np.array([sort_eigvec[i][4] for i in xrange(ngrid)])
ten_state = np.array([sort_eigvec[i][10] for i in xrange(ngrid)])
eleven = np.array([sort_eigvec[i][11] for i in xrange(ngrid)])
plt.xlim(min(x_grid) - 0.1, max(x_grid) + 0.1)
plt.ylim(min(ground) - 1.06, max(eleven) + 0.4)
plt.yticks([])
plt.plot(x_grid, eleven + 0.3, label='n = 11', color='black')
plt.plot(x_grid, ten_state, label='n = 10', color='purple')
plt.plot(x_grid, sec_state - 0.3, label='n = 2', color='green')
plt.plot(x_grid, first_state - 0.6, label='n = 1', color='blue')
plt.plot(x_grid, ground - 0.96, label='n = 0', color='dodgerblue')
plt.axhline(y = -0.96, linestyle='dashed', color='grey')
plt.axhline(y = -0.6, linestyle='dashed', color='grey')
plt.axhline(y = -0.3, linestyle='dashed', color='grey')
plt.axhline(y = 0, linestyle='dashed', color='grey')
plt.axhline(y = 0.3, linestyle='dashed', color='grey')
plt.xlabel("x")
plt.ylabel(r'$\psi_n(x)$')
plt.legend(loc='center right')
plt.savefig("morse_states.pdf")
plt.clf()

# Next 6 lines are solely for making plots of prob. densities
groundprob = np.array([sort_eigvec[i][0]*sort_eigvec[i][0] for i in xrange(ngrid)])
first_prob = np.array([sort_eigvec[i][1]*sort_eigvec[i][1] for i in xrange(ngrid)])
sec_prob = np.array([sort_eigvec[i][2]*sort_eigvec[i][2] for i in xrange(ngrid)])
third_prob = np.array([sort_eigvec[i][3]*sort_eigvec[i][3] for i in xrange(ngrid)])
fourth_prob = np.array([sort_eigvec[i][4]*sort_eigvec[i][4] for i in xrange(ngrid)])
ten_stateprob = np.array([sort_eigvec[i][10]*sort_eigvec[i][10] for i in xrange(ngrid)])
elevenprob = np.array([sort_eigvec[i][11]*sort_eigvec[i][11] for i in xrange(ngrid)])
plt.xlim(min(x_grid) - 0.01, max(x_grid) + 0.01)
plt.ylim(min(groundprob) - 0.106, max(elevenprob) + 0.04)
plt.yticks([])
plt.plot(x_grid, elevenprob + 0.03, label='n = 11', color='black')
plt.plot(x_grid, ten_stateprob, label='n = 10', color='purple')
plt.plot(x_grid, sec_prob - 0.03, label='n = 2', color='green')
plt.plot(x_grid, first_prob - 0.06, label='n = 1', color='blue')
plt.plot(x_grid, groundprob - 0.096, label='n = 0', color='dodgerblue')
plt.axhline(y = -0.096, linestyle='dashed', color='grey')
plt.axhline(y = -0.06, linestyle='dashed', color='grey')
plt.axhline(y = -0.03, linestyle='dashed', color='grey')
plt.axhline(y = 0, linestyle='dashed', color='grey')
plt.axhline(y = 0.03, linestyle='dashed', color='grey')
plt.xlabel("x")
plt.ylabel(r'$|\psi_n(x)|^2$')
plt.legend(loc='center right')
plt.savefig("morse_prob_densities.pdf")
plt.clf()

# Plotting the morse and harmonic potential.
potential_grid = np.linspace(-7.5, 15, 400)
harmonic_potential = np.array([0.5 * omegax**2 * 2 * d_well * i**2 for i in potential_grid])
morse_potential = np.array([d_well * (np.exp(-omegax * i) - 1)**2 for i in potential_grid])
plt.xlim(min(potential_grid) - 1, max(potential_grid) + 1)
plt.ylim(min(morse_potential), 18)
plt.plot(potential_grid, morse_potential, label='Morse Potential')
plt.plot(potential_grid, harmonic_potential, label="Harmonic Approximation", linestyle='dashed', color='grey')
plt.xlabel("x")
plt.ylabel("V(x)")
plt.legend()
plt.savefig("morse_potential.pdf")
plt.clf()

# Writing sorted eigenvalues to a file
f = open('1dmorse_eigenvalues.dat', 'wb')
f.write("n       E_n\n")
for i in xrange(ngrid):
    f.write("%s %s\n" % (i, sort_eigval[i]))
f.close()
