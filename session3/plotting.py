# numpy is a libarary used to do all kinds of mathematical operations
import numpy as np

# pandas allows us to organise data as tables (called "dataframes")
import pandas as pd

# we use plotly to make our figures
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px

# set the style of the plotly figures
pio.templates.default = "none"

# this allows us to make interactive figures
import ipywidgets as widgets

# import some custom fitting functions we wrote
from fitting import *

def plot_schedule(
    opt1Rewarded,
    trueProbability,
    magOpt1     = None,
    magOpt2     = None,
    probOpt1    = None,
    choiceProb1 = None,
    choice1     = None,
    utility1    = None,
    utility2    = None,
    ):
  '''
  Plots the experimental schedule and the RL model estimate using plotly.

    Parameters:
        opt1rewarded(int array): 1 if option 1 is rewarded on a trial, 0 if
          option 2 is rewarded on a trial.
        trueProbability(float array): The probability with which option 1 is
          rewareded on each trial.
        magOpt1(int array): The reward magnitude of option 1. Defaults to None,
          which excludes it from the plot.
        magOpt2(int array): The reward magnitude of option 1. Defaults to None,
          which excludes it from the plot.
        probOpt1(float array): how likely option 1 is rewarded on each trial
          according to the RL model. Defaults to None, which excludes it from
          the plot.
        choiceProb1(float array): the probability of choosing option 1 on each
          trial according to the RL model. Defaults to None, which excludes it
          from the plot.
        choice1(int array): whether option 1 (1) or option 2 (2) was chosen on
          each. Defaults to None, which excludes it from the plot.
        utility1(float array): the utility of option 1 on each trial according to
          the RL model. Defaults to None, which excludes it from the plot.
        utility2(float array): the utility of option 2 on each trial according to
          the RL model. Defaults to None, which excludes it from the plot.

  '''
  # compute number of trials
  nTrials = len(opt1Rewarded)

  # create 2 subplots if needed
  if magOpt1 is not None:
    fig = make_subplots(rows=2, cols=1)
  else:
    fig = go.Figure()

  # plot opt1rewarded as a scatterplot
  fig.add_trace(
      go.Scatter(
          x = list(range(nTrials)),
          y = opt1Rewarded,
          mode = 'markers',
          marker = dict(size = 5),
          name = "trial outcomes"
      ))

  # plot trueProbability as a line
  fig.add_trace(
      go.Scatter(
          x = list(range(nTrials)),
          y = trueProbability,
          mode = 'lines',
          line = dict(color='black', dash='dash'),
          name = "true probability"
      ))

  # check if we should plot probOpt1, and if so plot it as a line
  if probOpt1 is not None:
    fig.add_trace(
        go.Scatter(
            x = list(range(nTrials)),
            y = probOpt1,
            mode = 'lines',
            line = dict(color='red'),
            name = "RL model probability"
        ))

  # check if we should plot choiceProb1, and if so plot it as a scatterplot
  if choiceProb1 is not None:
    fig.add_trace(
        go.Scatter(
            x = list(range(nTrials)),
            y = choiceProb1,
            mode = 'markers',
            marker = dict(size = 5, symbol='x'),
            name = "choice probability"
        ))

  # check if we should plot choice1, and if so plot it as a scatterplot
  if choice1 is not None:
    fig.add_trace(
        go.Scatter(
            x = list(range(nTrials)),
            y = choice1,
            mode = 'markers',
            marker = dict(size = 5, symbol='cross', opacity=0.8),
            name = "simulated choice"
        ))

  # label the axes
  fig.update_layout(xaxis_title="", yaxis_title="green rewarded?")

  # check if we should plot reward magnitudes, and if so plot them as lines
  if magOpt1 is not None:
    fig.add_trace(
        go.Scatter(
            x = list(range(nTrials)),
            y = magOpt1,
            mode = 'lines',
            line = dict(color='orange'),
            name = "magnitude option 1",
            xaxis = 'x2',
            yaxis = 'y2',
        ))
    fig.add_trace(
        go.Scatter(
            x = list(range(nTrials)),
            y = magOpt2,
            mode = 'lines',
            line = dict(color='green'),
            name = "magnitude option 2",
            xaxis = 'x2',
            yaxis = 'y2',
        ))

    if utility1 is not None:
            
      # plot the utility of option 1 and option 2
      fig.add_trace(
          go.Scatter(
              x = list(range(nTrials)),
              y = utility1,
              mode = 'lines',
              line = dict(color='magenta'),
              name = "utility option 1",
              xaxis = 'x2',
              yaxis = 'y2',
          ))
      fig.add_trace(
          go.Scatter(
              x = list(range(nTrials)),
              y = utility2,
              mode = 'lines',
              line = dict(color='purple'),
              name = "utility option 2",
              xaxis = 'x2',
              yaxis = 'y2',
          ))
    # label the axes
    fig.update_layout(xaxis2_title="trial number", yaxis2_title="reward magnitude")

  # set x-axis range for all subplots
  # this sets the range for all x-axes
  fig.update_xaxes(range=[0, nTrials])
  # this is needed to set the range for the second subplot if it exists
  if magOpt1 is not None:
    fig.update_xaxes(range=[0, nTrials], row=2, col=1)

  return fig

def visualise_alpha_difference(stableAlphas, volatileAlphas, title):
  fig = px.histogram(pd.DataFrame({"stable block": stableAlphas,
                                  "volatile block": volatileAlphas}).melt(),
                                  color="variable", x="value", marginal="box", barmode="overlay")

  fig.update_layout(title= title, xaxis_title="learning rate", legend_title_text="")
  fig.update_xaxes(range=[0, 1])

  fig.show()