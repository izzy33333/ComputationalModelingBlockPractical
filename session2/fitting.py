# numpy is a libarary used to do all kinds of mathematical operations
import numpy as np

# we use the minimize function from scipy to fit models
from scipy.optimize import minimize

# import the logistic function
from scipy.stats import logistic

def utility_fun(mag, prob):
  return mag*prob

def loglikelihood_RL_model(opt1Rewarded,
                           magOpt1,
                           magOpt2,
                           choice1,
                           alpha,
                           beta,
                           *additonalParameters,
                           startingProb = 0.5,
                           utility_function = utility_fun,
                           choice_function = logistic.cdf):
  '''
  Returns the log likelihiood of the data given the choices and the model

    Parameters:
        opt1rewarded(bool array): True if option 1 is rewarded on a trial, False
          if option 2 is rewarded on a trial.
        magOpt1(int array): reward points between 1 and 100 for option 1 on each
          trial
        magOpt2(int array): reward points between 1 and 100 for option 2 on each
           trial
        alpha(float): fixed learning rate, greater than 0, less than/equal to 1
        beta(float): fixed inverse temperature, greater than 0
        *additionalParameters(float, optional): other parameters to pass onto
          the utility function
        startingProb(float): starting probability (defaults to 0.5).
        utility_function(function): what utility function to use to combine
          reward magnitude and probability. Defaults to multiplicative_utility
        choice_function(function): what choice function to use to decide
          between utility1 and utility2. Has free parameter beta. Defaults
          to softmax.

    Returns:
        LL(float): total log likelihood of the data given the input parameters
  '''

  nTrials = len(opt1Rewarded)

  # initialise some vectors we're going to assign into
  probOpt1 = np.zeros(nTrials, dtype = float)
  delta    = np.zeros(nTrials, dtype = float)

  # set the first trial's prediction to be equal to the starting probability
  probOpt1[0] = startingProb

  # initialise tracking the log likelhood
  LL = 0

  for t in range(nTrials-1):
        utility1 = utility_function(magOpt1[t], probOpt1[t], *additonalParameters)
        utility2 = utility_function(magOpt2[t], (1 - probOpt1[t]), *additonalParameters)
        if choice1[t] == 1:
          LL += np.log(choice_function((utility1-utility2)*beta))
        else:
          LL += np.log(choice_function((utility2-utility1)*beta))
        delta[t] = opt1Rewarded[t] - probOpt1[t]
        probOpt1[t+1] = probOpt1[t] + alpha*delta[t]

  return LL


def loglikelihood_trajectory(
                        opt1Rewarded,
                        magOpt1,
                        magOpt2,
                        choice1
                    ):
    '''
    Returns the log likelihiood of the data given the choices and the model for
        every step a Nelder-Mead solver takes on the likelihood landscape

    Parameters:
        opt1rewarded(bool array): True if option 1 is rewarded on a trial, False
          if option 2 is rewarded on a trial.
        magOpt1(int array): reward points between 1 and 100 for option 1 on each
          trial
        magOpt2(int array): reward points between 1 and 100 for option 2 on each
           trial
        choice1(bool array): True if option 1 was chosen on a trial, False if
          option 2 was chosen on a trial.
    '''
   
    # create function to be minimized
    def min_fun(x):
        '''
        Here we define a temporary function that we can pass to the minimizer.
        The minimze() function requires its first argument to be a function with
        one argument, which is why x is given to the likelihood function as a vector
        containing parameters x[0] (learning rate) and x[1] (inverse temperature). We
        also constrain the learning rate and inverse temperature by transforming the
        input parameters using functions that map [-Inf, Inf] ->  [0, 1] and
        [-Inf, Inf] -> [0, Inf] respectivly.
        '''
        return -loglikelihood_RL_model(opt1Rewarded,
                                        magOpt1,
                                        magOpt2,
                                        choice1,
                                        logistic.cdf(x[0]),
                                        np.exp(x[1]))

    # fit the data of this simulated participant
    recovered_parameters = minimize(min_fun, # the function we want to minimise
                                    [2, -0.5], # inital values for alpha and beta that the algorithm uses
                                    method = 'Nelder-Mead', # what minimisation algorithm to use
                                    options = {"return_all": True}) # this outputs every step the solver takes, which allows us to plot it


    niter = len(recovered_parameters.allvecs)
    alphas = np.zeros(niter)
    betas = np.zeros(niter)
    loglikelihoods = np.zeros(niter)
    for i in range(niter):
        alphas[i] = logistic.cdf(recovered_parameters.allvecs[i][0])
        betas[i] = np.exp(recovered_parameters.allvecs[i][1])
        loglikelihoods[i] = loglikelihood_RL_model(opt1Rewarded, magOpt1, magOpt2, choice1, alphas[i], betas[i])

    return alphas, betas, loglikelihoods