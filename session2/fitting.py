# numpy is a libarary used to do all kinds of mathematical operations
import numpy as np

# pandas allows us to organise data as tables (called "dataframes")
import pandas as pd

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
                           *additionalParameters,
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
        utility1 = utility_function(magOpt1[t], probOpt1[t], *additionalParameters)
        utility2 = utility_function(magOpt2[t], (1 - probOpt1[t]), *additionalParameters)
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
                        choice1,
                        method = 'Nelder-Mead' 
                    ):
    '''
    Returns the log likelihiood of the data given the choices and the model for
        every step a solver takes on the likelihood landscape

    Parameters:
        opt1rewarded(bool array): True if option 1 is rewarded on a trial, False
          if option 2 is rewarded on a trial.
        magOpt1(int array): reward points between 1 and 100 for option 1 on each
          trial
        magOpt2(int array): reward points between 1 and 100 for option 2 on each
           trial
        choice1(bool array): True if option 1 was chosen on a trial, False if
          option 2 was chosen on a trial.

    Returns:
        alphas(float array): learning rate at each step of the solver
        betas(float array): inverse temperature at each step of the solver
        loglikelihoods(float array): log likelihood at each step of the solver
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
                                    method = method, # what minimisation algorithm to use
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

def run_paramterer_recovery(
                        simulatedAlphaRange, 
                        simulatedBetaRange,
                        simulate_RL_model,
                        generate_schedule,
                        trueProbability,
                        rng,
                        method = 'Nelder-Mead'
                        ):
    '''
    This function simulates participants with different learning rates and then fits the data . This allows us to see if the model can recover the true learning rates.
    
    Parameters:
        simulatedAlphaRange(float array): alpha values to simulate
        simulatedBetaRange(float array): beta values to simulate
        simulate_RL_model(function): the function we use to simulate the RL model
        generate_schedule(function): the function we use to generate a schedule
        trueProbability(float array): the reward probabilities
        rng(random number generator): the random number generator we use

    Returns:
        recoveryData(dataframe): a dataframe containing the simulated and recovered parameters
    '''

    nSimulatedSubjects = len(simulatedAlphaRange) * len(simulatedBetaRange)

    # initialise a table to store the simualted and recoverd parameters in
    recoveryData = pd.DataFrame(np.zeros((nSimulatedSubjects, 4)),
                                columns = ["simulatedAlpha",
                                        "simulatedBeta",
                                        "recoveredAlpha",
                                        "recoveredBeta"])

    # initialize the iteration counter
    counter = 0

    for alpha in range(len(simulatedAlphaRange)):
        for beta in range(len(simulatedBetaRange)):
            if counter % 10 == 0:
                display("simulating and fitting subject " + str(counter) + "/" + str(nSimulatedSubjects))

            # generate a new schedule
            opt1Rewarded, magOpt1, magOpt2 = generate_schedule(trueProbability)

            # simulate an artificial participant
            probOpt1, choiceProb1 = simulate_RL_model(opt1Rewarded, magOpt1, magOpt2, simulatedAlphaRange[alpha], simulatedBetaRange[beta])
            choice1 = (choiceProb1 > rng.random(len(opt1Rewarded))).astype(int)

            # create function to be minimized
            def min_fun(x):
                return -loglikelihood_RL_model(opt1Rewarded, magOpt1, magOpt2, choice1, logistic.cdf(x[0]), np.exp(x[1]))

            # fit the data of this simulated participant
            recovered_parameters = minimize(min_fun, [0, -1.5], method = method)

            # save the data of the current iteration
            recoveryData.loc[counter,"simulatedAlpha"] = simulatedAlphaRange[alpha]
            recoveryData.loc[counter,"simulatedBeta"]  = simulatedBetaRange[beta]
            recoveryData.loc[counter,"recoveredAlpha"] = logistic.cdf(recovered_parameters.x[0])
            recoveryData.loc[counter,"recoveredBeta"]  = np.exp(recovered_parameters.x[1])

            # increase the iteration counter
            counter += 1

    return recoveryData


def run_paramterer_recovery_with_difference(
                stableAlphas,           
                volatileAlphas,         
                betas,                  
                simulate_RL_model,      
                generate_schedule,       
                trueProbabilityStable,   
                trueProbabilityVolatile, 
                rng,
                method = 'Nelder-Mead'                   
                ):
    ''' 
    This function simulates participants with different learning rates in the stable and volatile conditions, and then fits the data with a model that assumes the same learning rate in both conditions. This allows us to see if the model can recover the true learning rates in the stable and volatile conditions.
    
    Parameters:
        stableAlphas(float array): alpha values in the stable condition
        volatileAlphas(float array): alpha values in the volatile condition
        betas(float array): beta values in both conditions
        simulate_RL_model(function): the function we use to simulate the RL model
        generate_schedule(function): the function we use to generate a schedule
        trueProbabilityStable(float array): the reward probabilities in the stable condition
        trueProbabilityVolatile(float array): the reward probabilities in the volatile condition
        rng(random number generator): the random number generator we use

    Returns:
        fittedParameters(dataframe): a dataframe containing the recovered parameters
    '''

    nSimulatedSubjects = len(betas)

    # initialise a table to store the simualted and recoverd parameters in
    fittedParameters = pd.DataFrame(np.zeros((nSimulatedSubjects, 3)),
                                        columns = ["alpha stable",
                                                    "alpha volatile",
                                                    "inverse temperature"])

    for p in range(nSimulatedSubjects):
        if p % 10 == 0:
            display("fitting subject " + str(p) + "/" + str(nSimulatedSubjects))

        # generate a new schedule
        opt1RewardedStable, magOpt1Stable, magOpt2Stable = generate_schedule(trueProbabilityStable)
        opt1RewardedVolatile, magOpt1Volatile, magOpt2Volatile = generate_schedule(trueProbabilityVolatile)

        # simulate an artificial participant
        probOpt1Stable, choiceProb1Stable = simulate_RL_model(opt1RewardedStable, magOpt1Stable, magOpt2Stable, stableAlphas[p], betas[p])
        probOpt1Volatile, choiceProb1Volatile = simulate_RL_model(opt1RewardedVolatile, magOpt1Volatile, magOpt2Volatile, volatileAlphas[p], betas[p])
        choice1Stable = (choiceProb1Stable > rng.random(len(opt1RewardedStable))).astype(int)
        choice1Volatile = (choiceProb1Volatile > rng.random(len(opt1RewardedVolatile))).astype(int)

        # create function to be minimized
        def min_fun(x):
            LL1 = loglikelihood_RL_model(opt1RewardedStable, magOpt1Stable, magOpt2Stable, choice1Stable, logistic.cdf(x[0]), np.exp(x[2]))
            LL2 = loglikelihood_RL_model(opt1RewardedVolatile, magOpt1Volatile, magOpt2Volatile, choice1Volatile, logistic.cdf(x[1]), np.exp(x[2]))
            return -(LL1 + LL2)

        # fit the data of this simulated participant
        pars = minimize(min_fun, [0, 0, -1.5], method = method)

        fittedParameters.loc[p,"alpha stable"] = logistic.cdf(pars.x[0])
        fittedParameters.loc[p,"alpha volatile"] = logistic.cdf(pars.x[1])
        fittedParameters.loc[p,"inverse temperature"] = np.exp(pars.x[2])

    return fittedParameters