# numpy is a libarary used to do all kinds of mathematical operations
import numpy as np

# pandas allows us to organise data as tables (called "dataframes")
import pandas as pd

# we use the minimize function from scipy to fit models
from scipy.optimize import minimize

# import the logistic function
from scipy.stats import logistic

def multiplicative_utility(mag, prob):
  return mag*prob

def additive_utility(mag, prob, phi):
  return phi*prob + (1-phi)*(mag/80)

def loglikelihood_RL_model(opt1Rewarded,
                           magOpt1,
                           magOpt2,
                           choice1,
                           alpha,
                           beta,
                           *additonalParameters,
                           startingProb = 0.5,
                           utility_function = multiplicative_utility,
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