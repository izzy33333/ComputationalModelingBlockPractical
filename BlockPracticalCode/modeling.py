import numpy as np
from scipy.stats import logistic

def multiplicative_utility(mag,prob):
  return mag*prob

def additive_utility(mag,prob,omega):
  return (omega*(mag/100) + (1-omega)*prob)*100

def simulate_RL_model(opt1Rewarded,
                      magOpt1,
                      magOpt2,
                      alpha,
                      beta,
                      *additonalParameters,
                      startingProb = 0.5,
                      utility_function = multiplicative_utility,
                      choice_function = logistic.cdf):
  '''
  Returns how likely option 1 is rewarded on each trial, the probability of
  choosing option 1 on a trial, and a simulated choice for each trial

    Parameters:
        opt1rewarded(int array): 1 if option 1 is rewarded on a trial, 0 if option 2 is rewarded on a trial.
        
        magOpt1(int array): reward points between 1 and 100 for option 1 on each trial
        
        magOpt2(int array): reward points between 1 and 100 for option 2 on each trial
        
        alpha(float): fixed learning rate, greater than 0, less than/equal to 1
        
        beta(float): fixed inverse temperature, greater than 0
        
        *additionalParameters(float, optional): other parameters to pass onto the utility function
        
        startingProb(float): starting probability (defaults to 0.5).
        
        utility_function(function): what utility function to use to combine reward magnitude and probability. Defaults to multiplicative_utility
        
        choice_function(function): what choice function to use to decide between utility1 and utility2. Has free parameter beta. Defaults to sigmoid.

    Returns:
        probOpt1(float array): how likely option 1 is rewarded on each trial according to the RL model.
        
        choiceProb1(float array): the probability of choosing option 1 on each trial when combining infrmation about magnitude and probability
  '''

  # check that alpha has been set appropriately
  assert alpha > 0, 'Learning rate (alpha) must be greater than 0'
  assert alpha <= 1,'Learning rate (alpha) must be less than or equal to 1'

  # check that inverse temperateure has been set appropriately
  assert beta >= 0, 'beta must be greater or equal than 0'

  # check that startingProb has been set appropriately
  assert startingProb >= 0, 'startingProb must be greater or equal than 0'
  assert startingProb <= 1, 'startingProb must be less than or equal to 1'

  # calculate the number of trials
  nTrials = len(opt1Rewarded)

  # pre-create some vectors we're going to assign into
  probOpt1    = np.zeros(nTrials, dtype = float)
  delta       = np.zeros(nTrials, dtype = float)
  choiceProb1 = np.zeros(nTrials, dtype = float)

  # set the first trial's prediction to be equal to the starting probability
  probOpt1[0] = startingProb

  for t in range(nTrials-1):
        if magOpt1 is not None:
            # calculate the utility of the two options
            utility1 = utility_function(magOpt1[t], probOpt1[t], *additonalParameters)
            utility2 = utility_function(magOpt2[t], (1 - probOpt1[t]), *additonalParameters)
            # get the probability of making choice 1
            choiceProb1[t] = choice_function((utility1 - utility2) * beta)
        # calculate the prediction error
        delta[t] = opt1Rewarded[t] - probOpt1[t]
        # update the probability of option 1 being rewarded
        probOpt1[t+1] = probOpt1[t] + alpha*delta[t]

  choiceProb1[choiceProb1==1] = 0.999

  return probOpt1, choiceProb1

