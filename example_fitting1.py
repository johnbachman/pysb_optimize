from pysb.examples.robertson import model
import numpy as np
from matplotlib import pyplot as plt
from pysb.integrate import Solver
import scipy.optimize
from pyDOE import *
from scipy.stats.distributions import norm
import numdifftools as nd

# Simulate the model

# Create time vector using np.linspace function
num_timepoints = 101
num_dim = 3
t = np.linspace(0, 200, num_timepoints)

# Get instance of the Solver
sol = Solver(model, t, use_analytic_jacobian=False)

# Perform the integration
sol.run()

# sol.y contains timecourses for all of the species
# (model.species gives matched list of species)

# sol.yobs contains timecourse only for the observables
# Indexed by the name of the observable

# Plot the timecourses for A, B, and C
plt.ion()
plt.figure()

# Iterate over the observables
colors = ['red', 'green', 'blue']
data = np.zeros((num_timepoints, len(model.observables)))
for obs_ix, obs in enumerate(model.observables):
    obs_max = np.max(sol.yobs[obs.name])
    # Plot the observable
    plt.plot(t, sol.yobs[obs.name] / obs_max, label=obs.name,
             color=colors[obs_ix])
    # Make noisy data
    # Get random numbers
    rand_norm = np.random.randn(num_timepoints)
    # Multiply by the values of the observable
    sigma = 0.1
    noise = rand_norm * sigma * sol.yobs[obs.name]
    # Add the noise vector to the timecourse
    noisy_obs = noise + sol.yobs[obs.name]
    norm_noisy_data = noisy_obs / obs_max
    plt.plot(t, norm_noisy_data, linestyle='', marker='.',
             color=colors[obs_ix])
    data[:, obs_ix] = noisy_obs

p_to_fit = [p for p in model.parameters
                     if p.name in ['k1', 'k2', 'k3']]

num_calls = 0

# First define the objective function
def obj_func(x):
    global num_calls
    num_calls += 1
    lin_x = 10 ** x
    #print x
    # Run a simulation using these parameters
    # Initialize the model to have the values in the parameter array
    for p_ix, p in enumerate(p_to_fit):
        p.value = lin_x[p_ix]
    # Run the solver
    sol.run()
    # Calculate our error
    total_err = 0
    for obs_ix, obs in enumerate(model.observables):
        y = sol.yobs[obs.name]
        # Calculate the square difference with the data
        total_err += np.sum((y - data[:, obs_ix])**2)
    return total_err

# Hang on to the original values for comparison
nominal_values = np.array([p.value for p in p_to_fit])
x_test = np.log10(nominal_values)
#print "True values (in log10 space):", x_test
#print "Nominal error:", obj_func(x_test)




# Pick a starting point; in practice this would be random selected by
# a sampling strategy (e.g., latin hypercube sampling) or from a prior
# distributionmeans
x0 = np.array([np.log10(p.value * 0.1) for p in p_to_fit])

# Run the minimization algorithm!

def Jacob(x):
    #x = np.asarray(x)
    jaco = nd.Jacobian(obj_func)(x)
    #print jaco,'aaa'
    return jaco[0]

def Hessi(x):
    #x = np.asarray(x)
    hes = nd.Hessian(obj_func)(x)
    #print hes
    return hes

#Jacob = nda.Jacobian(obj_func, method = 'reverse')
#Hessi = nda.Hessian(obj_func)
#result = scipy.optimize.minimize(obj_func, x0, method='trust-ncg',jac = Jacob,
#                                 hess=Hessi)
result = scipy.optimize.minimize(obj_func, x0, method='nelder-mead')


plt.figure()
# Plot the original data
plt.plot(t, data, linestyle='', marker='.', color='k')
# Plot BEFORE
# Set parameter values to start position
for p_ix, p in enumerate(p_to_fit):
    p.value = 10 ** x0[p_ix]
sol.run()
plt.plot(t, sol.y, color='red')
# Plot AFTER
# Set parameter values to final position
for p_ix, p in enumerate(p_to_fit):
    p.value = 10 ** result.x[p_ix]
sol.run()
plt.plot(t, sol.y, color='yellow')


print "Num calls:", num_calls


