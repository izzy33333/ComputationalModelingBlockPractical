# numpy is a libarary used to do all kinds of mathematical operations
import numpy as np

# pandas allows us to organise data as tables (called "dataframes")
import pandas as pd

# load in the dataset
from scipy.io import loadmat
blain_data = loadmat("Blain_MoodTracksLearning_data.mat")

def load_blain(ID, data = blain_data):

  if ID < 37:
    order = "stable2volatile_data"
  else:
    order = "volatile2stable_data"
    ID -= 37

  trueProbability = data[order]["main"][0][ID]["task_mu"][0][0][0]
  outcome = data[order]["main"][0][ID]["maintask"][0][0][:,14] > 0
  numTrials = len(outcome)
  choice1       = np.empty(numTrials, dtype = int)
  magOpt1       = np.empty(numTrials, dtype = int)
  magOpt2       = np.empty(numTrials, dtype = int)
  opt1Rewarded  = np.empty(numTrials, dtype = int)
  
  for i in range(numTrials):
    choice1[i]  = 2 - data[order]["main"][0][ID]["list_pair"][0][0][int(data[order]["main"][0][ID]["maintask"][0][0][i,4]-1),i]
    magOpt1[i]  = data[order]["main"][0][ID]["mat_mag"][0][0][data[order]["main"][0][ID]["index_mag"][0][0][:,i][0]-1,0]
    magOpt2[i]  = data[order]["main"][0][ID]["mat_mag"][0][0][data[order]["main"][0][ID]["index_mag"][0][0][:,i][0]-1,1]
    if choice1[i]:
      if outcome[i]:
        opt1Rewarded[i] = 1
      else:
        opt1Rewarded[i] = 0
    else:
      if outcome[i]:
        opt1Rewarded[i] = 0
      else:
        opt1Rewarded[i] = 1

  return trueProbability, choice1, magOpt1, magOpt2, opt1Rewarded

def load_model_fits():
  data1AlphaMul = pd.read_csv("data1AlphaMul.csv")
  data2AlphaMul = pd.read_csv("data2AlphaMul.csv")
  data1AlphaAdd = pd.read_csv("data1AlphaAdd.csv")
  data2AlphaAdd = pd.read_csv("data2AlphaAdd.csv")
  return data1AlphaMul, data2AlphaMul, data1AlphaAdd, data2AlphaAdd