from pysb.examples.robertson import model
import numpy as np
from matplotlib import pyplot as plt
from pysb.integrate import Solver
import scipy.optimize
import scipy.stats as stats
from pyDOE import *
from scipy.stats.distributions import norm
import numdifftools as nd

# Simulate the model

# Create time vector using np.linspace function
num_timepoints = 101
num_dim = 3
t = np.linspace(0, 200, num_timepoints)

# Get instance of the Solver
sol = Solver(model, t, use_analytic_jacobian=True,integrator='lsoda')

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
colors = ['red', 'green', 'blue', 'yellow','black','darkviolet','cyan','magenta','darkgray','sienna']
data = np.zeros((num_timepoints, len(model.observables)))
for obs_ix, obs in enumerate(model.observables):
    obs_max = np.max(sol.yobs[obs.name])
    # Plot the observable
    plt.plot(t, sol.yobs[obs.name]/obs_max, label=obs.name, 
             color=colors[obs_ix])
    # Make noisy data
    # Get random numbers
    rand_norm = np.random.randn(num_timepoints)
    # Multiply by the values of the observable
    sigma = 0.1
    noise = rand_norm * sigma * sol.yobs[obs.name]
    # Add the noise vector to the timecourse
    noisy_obs = noise + sol.yobs[obs.name]
    norm_noisy_data = noisy_obs/obs_max
    plt.plot(t, norm_noisy_data, linestyle='', marker='.',
             color=colors[obs_ix])
    data[:, obs_ix] = noisy_obs

p_to_fit = [p for p in model.parameters
                     if p.name in ['k1', 'k2', 'k3']]


# First define the objective function
def obj_func(x):
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
 
    if np.isnan(total_err):
        print "Err is Nan"
        return np.inf		
    else:
    #print total_err
        return total_err

def Jacob(x):
	jaco = nd.Jacobian(obj_func)(x)
	return jaco[0]

def Hessi(x):
	hes = nd.Hessian(obj_func)(x)
	return hes


# Hang on to the original values for comparison
nominal_values = np.array([p.value for p in p_to_fit])
x_test = np.log10(nominal_values)

print "True values (in log10 space):", x_test
print "Nominal error:", obj_func(x_test)







# Pick a starting point; in practice this would be random selected by
# a sampling strategy (e.g., latin hypercube sampling) or from a prior
# distribution
num_rand = 6
design = lhs(len(p_to_fit), samples= num_rand/len(p_to_fit))
means = x_test
stdvs = np.array([0.1, 0.1, 0.1])
for alp in range(len(p_to_fit)):
	design[:,alp] = norm(loc=means[alp] , scale=stdvs[alp]).ppf(design[:,alp])

#Create a list of methods

meth_list = ['Nelder-Mead', 'Powell', 'COBYLA','TNC','L-BFGS-B','CG','BFGS','SLSQP','trust-ncg','Newton-CG']
met_list = ['dogleg']

#Create arrays for storing no. of function evaluations and objective function values

func_eval = np.zeros((len(meth_list),num_rand/len(p_to_fit)))
obj_val = np.zeros((len(meth_list),num_rand/len(p_to_fit)))

#Run the minimization algorithm for each initial value
#Store the objective function value and the no. of function evaluations made
for j in range(num_rand/len(p_to_fit)):
	
	x0 = design[j]
	for k, meth in enumerate(meth_list):
		if meth == 'trust-ncg' or meth == 'Newton-CG':
			result = scipy.optimize.minimize(obj_func, x0, method=meth,jac = Jacob, hess = Hessi)
			some = model.parameters
			print meth,j+1,some
		
			obj_val[k][j] = obj_func(result.x)
			func_eval[k][j] = result.nfev
		else:
			result = scipy.optimize.minimize(obj_func, x0, method=meth)
		
			some = model.parameters
			print meth,j+1,some
		
			obj_val[k][j] = obj_func(result.x)
			func_eval[k][j] = result.nfev
		

for q, meth in enumerate(meth_list):
	srt = sorted(func_eval[q])
	fit = stats.norm.pdf(srt, np.mean(srt), np.std(srt))  #this is a fitting 
	plt.figure()
	plt.plot(srt,fit,'-o')
	plt.hist(srt,normed=True,color = colors[q])      #draw histogram of data
	plt.show()            

lik = np.arange(num_rand/len(p_to_fit))
#mark = ['o','^','s','*','p','D','h','8']
plt.figure()
for ind, meth in enumerate(meth_list):
		plt.plot(lik+1,sorted(obj_val[ind]),marker='o',color = colors[ind],label=meth)
		plt.yscale('log')
		plt.legend()
		plt.xlabel("Run index")
		plt.ylabel("Objective function value (log)")
		plt.grid(True)

plt.figure()
for idx, meth in enumerate(meth_list):
		plt.plot(lik+1,sorted(func_eval[idx]),marker='o',color = colors[idx],label=meth)
		#plt.yscale('log')
		plt.legend()
		plt.xlabel("Run index")
		plt.ylabel("Calls to the objective function")
		plt.grid(True)
			
#Print objective function value array and function evaluation array

print func_eval
print obj_val	


"""plt.figure()
# Plot the original data
plt.plot(t, data, linestyle='', marker='o', color='k',linestyle='')
# Plot BEFORE
# Set parameter values to start position
for p_ix, p in enumerate(p_to_fit):
    p.value = 10 ** x0[p_ix]
sol.run()
plt.plot(t, sol.y, color='r')
# Plot AFTER
# Set parameter values to final position
for p_ix, p in enumerate(p_to_fit):
    p.value = 10 ** result.x[p_ix]
sol.run()
plt.plot(t, sol.y, color='magenta')

for p_ix, p in enumerate(p_to_fit):
    p.value = 10 ** result_2.x[p_ix]
sol.run()
plt.plot(t, sol.y, color='yellow') """



