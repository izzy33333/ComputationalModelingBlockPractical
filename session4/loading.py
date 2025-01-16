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
  data1AlphaMul = pd.read_csv("session4/data1AlphaMul.csv")
  data2AlphaMul = pd.read_csv("session4/data2AlphaMul.csv")
  data1AlphaAdd = pd.read_csv("session4/data1AlphaAdd.csv")
  data2AlphaAdd = pd.read_csv("session4/data2AlphaAdd.csv")
  data1AlphaMul.ID = data1AlphaMul.ID.astype(int)
  data2AlphaMul.ID = data2AlphaMul.ID.astype(int)
  data1AlphaAdd.ID = data1AlphaAdd.ID.astype(int)
  data2AlphaAdd.ID = data2AlphaAdd.ID.astype(int)
  return data1AlphaMul, data2AlphaMul, data1AlphaAdd, data2AlphaAdd

def load_parameter_recovery():
  recov1AlphaMul = pd.read_csv("session4/recov1AlphaMul.csv")
  recov2AlphaMul = pd.read_csv("session4/recov2AlphaMul.csv")
  recov1AlphaAdd = pd.read_csv("session4/recov1AlphaAdd.csv")
  recov2AlphaAdd = pd.read_csv("session4/recov2AlphaAdd.csv")
  recov1AlphaMul.ID = recov1AlphaMul.ID.astype(int)
  recov2AlphaMul.ID = recov2AlphaMul.ID.astype(int)
  recov1AlphaAdd.ID = recov1AlphaAdd.ID.astype(int)
  recov2AlphaAdd.ID = recov2AlphaAdd.ID.astype(int)
  return recov1AlphaMul, recov2AlphaMul, recov1AlphaAdd, recov2AlphaAdd