# numpy is a libarary used to do all kinds of mathematical operations
import numpy as np

# pandas allows us to organise data as tables (called "dataframes")
import pandas as pd

# we use the minimize function from scipy to fit models
from scipy.optimize import minimize

# import the logistic function
from scipy.stats import logistic

# we are using the chi2 distribution for some statistical tests
from scipy.stats import chi2

# this function allows us to perform one sample t-tests
from scipy.stats import ttest_1samp

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


def fit_participant_data(utility_function, IDs, simulate = False, alpha_S = None, alpha_V = None, beta = None, omega = None, rng = np.random.default_rng(12345)):
  
  numSubjects = len(IDs)
  
  
  fitData1AlphaMul = pd.DataFrame(np.zeros((numSubjects, 5)),
                              columns = ["ID",
                                          "alpha",
                                          "beta",
                                          "LL",
                                          "BIC"])

  fitData2AlphaMul = pd.DataFrame(np.zeros((numSubjects, 6)),
                              columns = ["ID",
                                          "alphaStable",
                                          "alphaVolatile",
                                          "beta",
                                          "LL",
                                          "BIC"])
  
  fitData1AlphaAdd = pd.DataFrame(np.zeros((numSubjects, 6)),
                              columns = ["ID",
                                          "alpha",
                                          "beta",
                                          "omega",
                                          "LL",
                                          "BIC"])

  fitData2AlphaAdd = pd.DataFrame(np.zeros((numSubjects, 7)),
                              columns = ["ID",
                                          "alphaStable",
                                          "alphaVolatile",
                                          "beta",
                                          "omega",
                                          "LL",
                                           "BIC"])


  for i, s in enumerate(IDs):    

    # load in data
    _, choice1, magOpt1, magOpt2, opt1Rewarded = load_blain(s)
    
    if simulate:
      if utility_function == multiplicative_utility:  
        # simulate an artificial participant
        if s < 37:
          choice1[0:80], _, _, _, _   = simulate_RL_model(opt1Rewarded[0:80],   magOpt1[0:80],   magOpt2[0:80],   alpha_S[s], beta[s], utility_function = utility_function, rng = rng)
          choice1[80:160], _, _, _, _ = simulate_RL_model(opt1Rewarded[80:160], magOpt1[80:160], magOpt2[80:160], alpha_V[s], beta[s], utility_function = utility_function, rng = rng)
        else:
          choice1[0:80], _, _, _, _   = simulate_RL_model(opt1Rewarded[0:80],   magOpt1[0:80],   magOpt2[0:80],   alpha_V[s], beta[s], utility_function = utility_function, rng = rng)
          choice1[80:160], _, _, _, _ = simulate_RL_model(opt1Rewarded[80:160], magOpt1[80:160], magOpt2[80:160], alpha_S[s], beta[s], utility_function = utility_function, rng = rng)
      elif utility_function == additive_utility:
        # simulate an artificial participant
        if s < 37:
          choice1[0:80], _, _, _, _   = simulate_RL_model(opt1Rewarded[0:80],   magOpt1[0:80],   magOpt2[0:80],   alpha_S[s], beta[s], omega[s], utility_function = utility_function, rng = rng)
          choice1[80:160], _, _, _, _ = simulate_RL_model(opt1Rewarded[80:160], magOpt1[80:160], magOpt2[80:160], alpha_V[s], beta[s], omega[s], utility_function = utility_function, rng = rng)
        else:
          choice1[0:80], _, _, _, _   = simulate_RL_model(opt1Rewarded[0:80],   magOpt1[0:80],   magOpt2[0:80],   alpha_V[s], beta[s], omega[s], utility_function = utility_function, rng = rng)
          choice1[80:160], _, _, _, _ = simulate_RL_model(opt1Rewarded[80:160], magOpt1[80:160], magOpt2[80:160], alpha_S[s], beta[s], omega[s], utility_function = utility_function, rng = rng)
          
        
    # create functions to be minimized
    def min_fun(x):
      LL1 = loglikelihood_RL_model(opt1Rewarded[0:80],   magOpt1[0:80],   magOpt2[0:80],   choice1[0:80],   logistic.cdf(x[0]), logistic.cdf(x[1]), utility_function = multiplicative_utility)
      LL2 = loglikelihood_RL_model(opt1Rewarded[80:160], magOpt1[80:160], magOpt2[80:160], choice1[80:160], logistic.cdf(x[0]), logistic.cdf(x[1]), utility_function = multiplicative_utility)
      return - (LL1 + LL2)

    # fit the data of this participant
    fitted_parameters_1_alpha_mul = minimize(min_fun, [0, -1.5], method = 'Nelder-Mead')
    
    fitData1AlphaMul.BIC[i]   = 2*np.log(160) + 2*fitted_parameters_1_alpha_mul.fun
    fitData1AlphaMul.alpha[i] = logistic.cdf(fitted_parameters_1_alpha_mul.x[0])
    fitData1AlphaMul.beta[i]  = logistic.cdf(fitted_parameters_1_alpha_mul.x[1])
    fitData1AlphaMul.LL[i]    = -fitted_parameters_1_alpha_mul.fun
    fitData1AlphaMul.ID[i]    = s
      
    # create functions to be minimized
    def min_fun(x):
      LL1 = loglikelihood_RL_model(opt1Rewarded[0:80],   magOpt1[0:80],   magOpt2[0:80],   choice1[0:80],   logistic.cdf(x[0]), logistic.cdf(x[1]), logistic.cdf(x[2]), utility_function = additive_utility)
      LL2 = loglikelihood_RL_model(opt1Rewarded[80:160], magOpt1[80:160], magOpt2[80:160], choice1[80:160], logistic.cdf(x[0]), logistic.cdf(x[1]), logistic.cdf(x[2]), utility_function = additive_utility)
      return - (LL1 + LL2)

    # fit the data of this participant
    fitted_parameters_1_alpha_add = minimize(min_fun, [0, -1.5, 0], method = 'BFGS')
      
    fitData1AlphaAdd.omega[i] = logistic.cdf(fitted_parameters_1_alpha_add.x[2])
    fitData1AlphaAdd.BIC[i] = 3*np.log(160) + 2*fitted_parameters_1_alpha_add.fun
    fitData1AlphaAdd.alpha[i] = logistic.cdf(fitted_parameters_1_alpha_add.x[0])
    fitData1AlphaAdd.beta[i]  = logistic.cdf(fitted_parameters_1_alpha_add.x[1])
    fitData1AlphaAdd.LL[i]    = -fitted_parameters_1_alpha_add.fun
    fitData1AlphaAdd.ID[i]    = s

    # create functions to be minimized
    def min_fun(x):
      LL1 = loglikelihood_RL_model(opt1Rewarded[0:80],   magOpt1[0:80],   magOpt2[0:80],   choice1[0:80],   logistic.cdf(x[0]), logistic.cdf(x[2]), utility_function = multiplicative_utility)
      LL2 = loglikelihood_RL_model(opt1Rewarded[80:160], magOpt1[80:160], magOpt2[80:160], choice1[80:160], logistic.cdf(x[1]), logistic.cdf(x[2]), utility_function = multiplicative_utility)
      return - (LL1 + LL2)

    # fit the data of this participant
    fitted_parameters_2_alpha_mul = minimize(min_fun, [fitted_parameters_1_alpha_mul.x[0], fitted_parameters_1_alpha_mul.x[0], fitted_parameters_1_alpha_mul.x[1]], method = 'Nelder-Mead')
    
    fitData2AlphaMul.BIC[i]  = 3*np.log(160) + 2*fitted_parameters_2_alpha_mul.fun
    if s < 37:
      fitData2AlphaMul.alphaStable[i]   = logistic.cdf(fitted_parameters_2_alpha_mul.x[0])
      fitData2AlphaMul.alphaVolatile[i] = logistic.cdf(fitted_parameters_2_alpha_mul.x[1])

    else:
      fitData2AlphaMul.alphaStable[i]   = logistic.cdf(fitted_parameters_2_alpha_mul.x[1])
      fitData2AlphaMul.alphaVolatile[i] = logistic.cdf(fitted_parameters_2_alpha_mul.x[0])

    fitData2AlphaMul.beta[i] = logistic.cdf(fitted_parameters_2_alpha_mul.x[2])
    fitData2AlphaMul.LL[i]   = -fitted_parameters_2_alpha_mul.fun
    fitData2AlphaMul.ID[i]   = s
      
    # create functions to be minimized
    def min_fun(x):
      LL1 = loglikelihood_RL_model(opt1Rewarded[0:80],   magOpt1[0:80],   magOpt2[0:80],   choice1[0:80],   logistic.cdf(x[0]), logistic.cdf(x[2]), logistic.cdf(x[3]), utility_function = additive_utility)
      LL2 = loglikelihood_RL_model(opt1Rewarded[80:160], magOpt1[80:160], magOpt2[80:160], choice1[80:160], logistic.cdf(x[1]), logistic.cdf(x[2]), logistic.cdf(x[3]), utility_function = additive_utility)
      return - (LL1 + LL2)

    # fit the data of this participant
    fitted_parameters_2_alpha_add = minimize(min_fun, [fitted_parameters_1_alpha_add.x[0], fitted_parameters_1_alpha_add.x[0], fitted_parameters_1_alpha_add.x[1], fitted_parameters_1_alpha_add.x[2]], method = 'BFGS')

    fitData2AlphaAdd.omega[i] = logistic.cdf(fitted_parameters_2_alpha_add.x[3])
    fitData2AlphaAdd.BIC[i] = 4*np.log(160) + 2*fitted_parameters_2_alpha_add.fun
    if s < 37:
      fitData2AlphaAdd.alphaStable[i]   = logistic.cdf(fitted_parameters_2_alpha_add.x[0])
      fitData2AlphaAdd.alphaVolatile[i] = logistic.cdf(fitted_parameters_2_alpha_add.x[1])

    else:
      fitData2AlphaAdd.alphaStable[i]   = logistic.cdf(fitted_parameters_2_alpha_add.x[1])
      fitData2AlphaAdd.alphaVolatile[i] = logistic.cdf(fitted_parameters_2_alpha_add.x[0])

    fitData2AlphaAdd.beta[i] = logistic.cdf(fitted_parameters_2_alpha_add.x[2])
    fitData2AlphaAdd.LL[i]   = -fitted_parameters_2_alpha_add.fun
    fitData2AlphaAdd.ID[i]   = s  
        
  return fitData1AlphaMul, fitData2AlphaMul, fitData1AlphaAdd, fitData2AlphaAdd


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

def parameter_recovery(
  data,
  nReps = 100,
  rng = np.random.default_rng(12345),
):
  dataOut = pd.DataFrame()
  for i in range(nReps):
    display("fitting iteration " + str(i+1) + " of " + str(nReps))
    dataTmp = data.copy()
    dataTmp = dataTmp.drop(columns=['LL', 'BIC'])
    dataTmp.beta = rng.permutation(dataTmp.beta)
    if any(dataTmp.columns == 'alphaStable'):
      dataTmp.alphaStable = rng.permutation(dataTmp.alphaStable)
      dataTmp.alphaVolatile = rng.permutation(dataTmp.alphaVolatile)
      if any(dataTmp.columns == 'omega'):
        dataTmp.omega = rng.permutation(dataTmp.omega)
        data1AlphaMul, data2AlphaMul, data1AlphaAdd, data2AlphaAdd = fit_participant_data(
          additive_utility,
          dataTmp.ID,
          simulate=True,
          alpha_S = dataTmp.alphaStable,
          alpha_V = dataTmp.alphaVolatile,
          beta = dataTmp.beta,
          omega = dataTmp.omega,
          rng = rng          
          )
      else:
        data1AlphaMul, data2AlphaMul, data1AlphaAdd, data2AlphaAdd = fit_participant_data(
          multiplicative_utility,
          dataTmp.ID,
          simulate=True,
          alpha_S = dataTmp.alphaStable,
          alpha_V = dataTmp.alphaVolatile,
          beta = dataTmp.beta,
          rng = rng
          )
    else:
      dataTmp.alpha = rng.permutation(dataTmp.alpha)
      if any(dataTmp.columns == 'omega'):
        dataTmp.omega = rng.permutation(dataTmp.omega)
        data1AlphaMul, data2AlphaMul, data1AlphaAdd, data2AlphaAdd = fit_participant_data(
          additive_utility,
          dataTmp.ID,
          simulate=True,
          alpha_S = dataTmp.alpha,
          alpha_V = dataTmp.alpha,
          beta = dataTmp.beta,
          omega = dataTmp.omega,
          rng = rng
          )
      else:
        data1AlphaMul, data2AlphaMul, data1AlphaAdd, data2AlphaAdd = fit_participant_data(
          multiplicative_utility,
          dataTmp.ID,
          simulate=True,
          alpha_S = dataTmp.alpha,
          alpha_V = dataTmp.alpha,
          beta = dataTmp.beta,
          rng = rng
          )
    dataTmp['recovered1MulAlpha'] = np.array(data1AlphaMul.alpha)
    dataTmp['recovered1MulBeta'] = np.array(data1AlphaMul.beta)
    dataTmp['recovered1MulLL'] = np.array(data1AlphaMul.LL)
    dataTmp['recovered1MulBIC'] = np.array(data1AlphaMul.BIC)
    dataTmp['recovered2MulAlphaS'] = np.array(data2AlphaMul.alphaStable)
    dataTmp['recovered2MulAlphaV'] = np.array(data2AlphaMul.alphaVolatile)
    dataTmp['recovered2MulBeta'] = np.array(data2AlphaMul.beta)
    dataTmp['recovered2MulLL'] = np.array(data2AlphaMul.LL)
    dataTmp['recovered2MulBIC'] = np.array(data2AlphaMul.BIC)
    
    dataTmp['recovered1AddAlpha'] = np.array(data1AlphaAdd.alpha)
    dataTmp['recovered1AddBeta'] = np.array(data1AlphaAdd.beta)
    dataTmp['recovered1AddOmega'] = np.array(data1AlphaAdd.omega)
    dataTmp['recovered1AddLL'] = np.array(data1AlphaAdd.LL)
    dataTmp['recovered1AddBIC'] = np.array(data1AlphaAdd.BIC)
    dataTmp['recovered2AddAlphaS'] = np.array(data2AlphaAdd.alphaStable)
    dataTmp['recovered2AddAlphaV'] = np.array(data2AlphaAdd.alphaVolatile)
    dataTmp['recovered2AddBeta'] = np.array(data2AlphaAdd.beta)
    dataTmp['recovered2AddOmega'] = np.array(data2AlphaAdd.omega)
    dataTmp['recovered2AddLL'] = np.array(data2AlphaAdd.LL)
    dataTmp['recovered2AddBIC'] = np.array(data2AlphaAdd.BIC)
    dataTmp.ID = dataTmp.ID + i*75
    dataOut = pd.concat([dataOut, dataTmp])
    
  return dataOut

def recov_chi2_test(recovData, degrees_of_freedom = 75, nReps = 100):
  p_values_mul = np.zeros(nReps)
  p_values_add = np.zeros(nReps)
  for i in range(nReps):
    lambda_LR = 2*sum(recovData['recovered2MulLL'][degrees_of_freedom*i:degrees_of_freedom+degrees_of_freedom*i] - recovData['recovered1MulLL'][degrees_of_freedom*i:degrees_of_freedom+degrees_of_freedom*i] )
    p_values_mul[i] = chi2.sf(lambda_LR, degrees_of_freedom)
    lambda_LR = 2*sum(recovData['recovered2AddLL'][degrees_of_freedom*i:degrees_of_freedom+degrees_of_freedom*i] - recovData['recovered1AddLL'][degrees_of_freedom*i:degrees_of_freedom+degrees_of_freedom*i] )
    p_values_add[i] = chi2.sf(lambda_LR, degrees_of_freedom)
  return p_values_mul, p_values_add

def recov_BICs(recovData, num_simulated_participants = 75, nReps = 100):
  winning_BIC = np.zeros(4)
  for i in range(nReps):
    winning_BIC[np.argmin([sum(recovData['recovered1MulBIC'][num_simulated_participants*i:num_simulated_participants+num_simulated_participants*i]),sum(recovData['recovered2MulBIC'][num_simulated_participants*i:num_simulated_participants+num_simulated_participants*i]),sum(recovData['recovered1AddBIC'][num_simulated_participants*i:num_simulated_participants+num_simulated_participants*i]),sum(recovData['recovered2AddBIC'][num_simulated_participants*i:num_simulated_participants+num_simulated_participants*i])])] += 1
  return winning_BIC

def recov_t_test(recovData, num_simulated_participants = 75, nReps = 100):
  p_values_mul = np.zeros(nReps)
  p_values_add = np.zeros(nReps)
  for i in range(nReps):
    res = ttest_1samp(recovData['recovered2MulAlphaV'][num_simulated_participants*i:num_simulated_participants+num_simulated_participants*i] - recovData['recovered2MulAlphaS'][num_simulated_participants*i:num_simulated_participants+num_simulated_participants*i], 0, alternative = "greater")
    p_values_mul[i] = res.pvalue
    res = ttest_1samp(recovData['recovered2AddAlphaV'][num_simulated_participants*i:num_simulated_participants+num_simulated_participants*i] - recovData['recovered2AddAlphaS'][num_simulated_participants*i:num_simulated_participants+num_simulated_participants*i], 0, alternative = "greater")
    p_values_add[i] = res.pvalue
  return p_values_mul, p_values_add