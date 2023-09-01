# we use plotly to make our figures
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px

# set the style of the plotly figures
pio.templates.default = "none"

# this allows us to make interactive figures
import ipywidgets as widgets

def plot_schedule(opt1Rewarded, trueProbability, probOpt1 = None):
  '''
  Plots the experimental schedule and the RL model estimate.

    Parameters:
        opt1rewarded(int array): 1 if option 1 is rewarded on a trial, 0 if
          option 2 is rewarded on a trial.
        trueProbability(float array): The probability with which option 1 is
          rewareded on each trial.
        probOpt1(float array): how likely option 1 is rewarded on each trial
          according to the RL model. Defaults to None, which excludes it from
          the plot.
  '''

  # compute number of trials
  nTrials = len(opt1Rewarded)

  # initalise figure
  fig = go.Figure()

  # plot opt1rewarded as a scatterplot
  fig.add_trace(
      go.Scatter(
          x = list(range(nTrials)),
          y = opt1Rewarded,
          mode = 'markers',
          marker = dict(color="blue", size = 5),
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

  # label the axes
  fig.update_layout(xaxis_title="trial number", yaxis_title="green rewarded?")

  # set the range for the axes
  fig.update_xaxes(range=[0, nTrials])
  fig.update_yaxes(range=[-0.05, 1.05])

  return fig
