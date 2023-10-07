# numpy is a libarary used to do all kinds of mathematical operations
import numpy as np

# pandas allows us to organise data as tables (called "dataframes")
import pandas as pd

# we use the minimize function from scipy to fit models
from scipy.optimize import minimize

# import the logistic function
from scipy.stats import logistic

from loading import *

def multiplicative_utility(mag, prob):
  return mag*prob

def additive_utility(mag, prob, omega):
  return (omega*(mag/100) + (1-omega)*prob)*100

def loglikelihood_RL_model(opt1Rewarded,
                          magOpt1,
                          magOpt2,
                          choice1,
                          alpha,
                          beta,
                          *additionalParameters,
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

  # set the first trial's prediction to be equal to the starting probability
  probOpt1 = startingProb

  # initialise tracking the log likelhood
  LL = 0

  for t in range(nTrials-1):
        utility1 = utility_function(magOpt1[t], probOpt1, *additionalParameters)
        utility2 = utility_function(magOpt2[t], (1 - probOpt1), *additionalParameters)
        if choice1[t] == 1:
          LL += np.log(choice_function((utility1-utility2)*beta))
        else:
          LL += np.log(choice_function((utility2-utility1)*beta))
        delta = opt1Rewarded[t] - probOpt1
        probOpt1 = probOpt1 + alpha * delta

  return LL


def fit_participant_data(utility_function, simulate = False, alpha_S = None, alpha_V = None, beta = None, rng = np.random.default_rng(12345), method = 'BFGS'):
  
  numSubjects = 75
  
  if utility_function == multiplicative_utility:
  
    fitData1Alpha = pd.DataFrame(np.zeros((numSubjects, 5)),
                                columns = ["ID",
                                           "alpha",
                                           "beta",
                                           "LL",
                                           "BIC"])

    fitData2Alpha = pd.DataFrame(np.zeros((numSubjects, 6)),
                                columns = ["ID",
                                           "alphaStable",
                                           "alphaVolatile",
                                           "beta",
                                           "LL",
                                           "BIC"])
  elif utility_function == additive_utility:
    
    fitData1Alpha = pd.DataFrame(np.zeros((numSubjects, 6)),
                                columns = ["ID",
                                           "alpha",
                                           "beta",
                                           "omega",
                                           "LL",
                                           "BIC"])

    fitData2Alpha = pd.DataFrame(np.zeros((numSubjects, 7)),
                                columns = ["ID",
                                           "alphaStable",
                                           "alphaVolatile",
                                           "beta",
                                           "omega",
                                           "LL",
                                           "BIC"])


  for s in range(numSubjects):
    if s % 5 == 0:
      display("fitting subject " + str(s) + "/" + str(numSubjects))
    

    # load in data
    trueProbability, choice1, magOpt1, magOpt2, opt1Rewarded = load_blain(s)
    
    if simulate:
      # simulate an artificial participant
      if s < 37:
        choice1[0:80], _, _, _, _   = simulate_RL_model(opt1Rewarded[0:80],   magOpt1[0:80],   magOpt2[0:80],   alpha_S[s], beta[s], rng = rng)
        choice1[80:160], _, _, _, _ = simulate_RL_model(opt1Rewarded[80:160], magOpt1[80:160], magOpt2[80:160], alpha_V[s], beta[s], rng = rng)
      else:
        choice1[0:80], _, _, _, _   = simulate_RL_model(opt1Rewarded[0:80],   magOpt1[0:80],   magOpt2[0:80],   alpha_V[s], beta[s], rng = rng)
        choice1[80:160], _, _, _, _ = simulate_RL_model(opt1Rewarded[80:160], magOpt1[80:160], magOpt2[80:160], alpha_S[s], beta[s], rng = rng)

    if utility_function == multiplicative_utility:
      # create functions to be minimized
      def min_fun(x):
        LL1 = loglikelihood_RL_model(opt1Rewarded[0:80],   magOpt1[0:80],   magOpt2[0:80],   choice1[0:80],   logistic.cdf(x[0]), np.exp(x[1]), utility_function = utility_function)
        LL2 = loglikelihood_RL_model(opt1Rewarded[80:160], magOpt1[80:160], magOpt2[80:160], choice1[80:160], logistic.cdf(x[0]), np.exp(x[1]), utility_function = utility_function)
        return - (LL1 + LL2)

      # fit the data of this participant
      fitted_parameters_1_alpha = minimize(min_fun, [0, -1.5], method = method)
      
      fitData1Alpha.BIC[s]   = 2*np.log(160) + 2*fitted_parameters_1_alpha.fun
      
    elif utility_function == additive_utility:
      # create functions to be minimized
      def min_fun(x):
        LL1 = loglikelihood_RL_model(opt1Rewarded[0:80],   magOpt1[0:80],   magOpt2[0:80],   choice1[0:80],   logistic.cdf(x[0]), np.exp(x[1]), logistic.cdf(x[2]), utility_function = utility_function)
        LL2 = loglikelihood_RL_model(opt1Rewarded[80:160], magOpt1[80:160], magOpt2[80:160], choice1[80:160], logistic.cdf(x[0]), np.exp(x[1]), logistic.cdf(x[2]), utility_function = utility_function)
        return - (LL1 + LL2)

      # fit the data of this participant
      fitted_parameters_1_alpha = minimize(min_fun, [0, -1.5, 0], method = method)
      
      fitData1Alpha.omega[s] = logistic.cdf(fitted_parameters_1_alpha.x[2])
      fitData1Alpha.BIC[s] = 3*np.log(160) + 2*fitted_parameters_1_alpha.fun

    # save the data
    fitData1Alpha.alpha[s] = logistic.cdf(fitted_parameters_1_alpha.x[0])
    fitData1Alpha.beta[s]  = np.exp(fitted_parameters_1_alpha.x[1])
    fitData1Alpha.LL[s]    = -fitted_parameters_1_alpha.fun
    fitData1Alpha.ID[s]    = s
    
    if utility_function == multiplicative_utility:  
      # create functions to be minimized
      def min_fun(x):
        LL1 = loglikelihood_RL_model(opt1Rewarded[0:80],   magOpt1[0:80],   magOpt2[0:80],   choice1[0:80],   logistic.cdf(x[0]), np.exp(x[2]), utility_function = utility_function)
        LL2 = loglikelihood_RL_model(opt1Rewarded[80:160], magOpt1[80:160], magOpt2[80:160], choice1[80:160], logistic.cdf(x[1]), np.exp(x[2]), utility_function = utility_function)
        return - (LL1 + LL2)

      # fit the data of this participant
      fitted_parameters_2_alpha = minimize(min_fun, [fitted_parameters_1_alpha.x[0], fitted_parameters_1_alpha.x[0], fitted_parameters_1_alpha.x[1]], method = method)
      
      fitData2Alpha.BIC[s]  = 3*np.log(160) + 2*fitted_parameters_2_alpha.fun
      
    elif utility_function == additive_utility:
      # create functions to be minimized
      def min_fun(x):
        LL1 = loglikelihood_RL_model(opt1Rewarded[0:80],   magOpt1[0:80],   magOpt2[0:80],   choice1[0:80],   logistic.cdf(x[0]), np.exp(x[2]), logistic.cdf(x[3]), utility_function = utility_function)
        LL2 = loglikelihood_RL_model(opt1Rewarded[80:160], magOpt1[80:160], magOpt2[80:160], choice1[80:160], logistic.cdf(x[1]), np.exp(x[2]), logistic.cdf(x[3]), utility_function = utility_function)
        return - (LL1 + LL2)

      # fit the data of this participant
      fitted_parameters_2_alpha = minimize(min_fun, [fitted_parameters_1_alpha.x[0], fitted_parameters_1_alpha.x[0], fitted_parameters_1_alpha.x[1], fitted_parameters_1_alpha.x[2]], method = method)

      fitData2Alpha.omega[s] = logistic.cdf(fitted_parameters_2_alpha.x[3])
      fitData2Alpha.BIC[s] = 4*np.log(160) + 2*fitted_parameters_2_alpha.fun
      
    # save the data
    if s < 37:
      fitData2Alpha.alphaStable[s]   = logistic.cdf(fitted_parameters_2_alpha.x[0])
      fitData2Alpha.alphaVolatile[s] = logistic.cdf(fitted_parameters_2_alpha.x[1])

    else:
      fitData2Alpha.alphaStable[s]   = logistic.cdf(fitted_parameters_2_alpha.x[1])
      fitData2Alpha.alphaVolatile[s] = logistic.cdf(fitted_parameters_2_alpha.x[0])

    fitData2Alpha.beta[s] = np.exp(fitted_parameters_2_alpha.x[2])
    fitData2Alpha.LL[s]   = -fitted_parameters_2_alpha.fun
    fitData2Alpha.ID[s]   = s
    
  return fitData1Alpha, fitData2Alpha


def simulate_RL_model(opt1Rewarded,
                      magOpt1,
                      magOpt2,
                      alpha,
                      beta,
                      *additionalParameters,
                      startingProb = 0.5,
                      utility_function = multiplicative_utility,
                      rng = np.random.default_rng(12345)):
  '''
  Returns how likely option 1 is rewarded on each trial, the probability of
  choosing option 1 on a trial, and a simulated choice for each trial

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
          the utility function, for example, the omega used in additive utility.
        startingProb(float): starting probability (defaults to 0.5).
        utility_function(function): what utility function to use to combine
          reward magnitude and probability. Defaults to multiplicative_utility
        choice_function(function): what choice function to use to decide
          between utility1 and utility2. Has free parameter beta. Defaults
          to softmax.

    Returns:
        probOpt1(float array): how likely option 1 is rewarded on each trial
          according to the RL model.
        choiceProb1(float array): the probability of choosing option 1 on each
          trial when combining infrmation about magnitude and probability
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
  utility1    = np.zeros(nTrials, dtype = float)
  utility2    = np.zeros(nTrials, dtype = float)

  # set the first trial's prediction to be equal to the starting probability
  probOpt1[0] = startingProb

  for t in range(nTrials-1):
        # calculate the utility of the two options. *additionalParameters would only be needed
        # if the utility function has >2 inputs, which is not the case for multiplicative
        # utility.
        utility1[t] = utility_function(magOpt1[t], probOpt1[t], *additionalParameters)
        utility2[t] = utility_function(magOpt2[t], (1 - probOpt1[t]), *additionalParameters)

        # get the probability of making choice 1
        choiceProb1[t] = logistic.cdf((utility1[t]-utility2[t])* beta)

        # calculate the prediction error
        delta[t] = opt1Rewarded[t] - probOpt1[t]

        # update the probability of option 1 being rewarded
        probOpt1[t+1] = probOpt1[t] + alpha*delta[t]
  
  t = nTrials-1
  utility1[t] = utility_function(magOpt1[t], probOpt1[t], *additionalParameters)
  utility2[t] = utility_function(magOpt2[t], (1 - probOpt1[t]), *additionalParameters)
  choiceProb1[t] = logistic.cdf((utility1[t]-utility2[t])* beta)
        
  choice1 = (choiceProb1 > rng.random(len(opt1Rewarded))).astype(int)

  return choice1, probOpt1, choiceProb1, utility1, utility2