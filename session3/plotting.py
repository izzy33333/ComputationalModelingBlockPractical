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

# import some custom functions we wrote
from fitting import *
from loading import *


def plot_schedule(ID):
  '''
  Plots the experimental schedule using plotly.

    Parameters:
        ID (int): The participant ID.

  '''
  
  trueProbability, choice1, magOpt1, magOpt2, opt1Rewarded = load_blain(ID)
  
  # compute number of trials
  nTrials = len(opt1Rewarded)

  # create 2 subplots
  fig = make_subplots(rows=2, cols=1)

  # plot trueProbability as a line
  fig.add_trace(
      go.Scatter(
          x = list(range(nTrials)),
          y = trueProbability,
          mode = 'lines',
          line = dict(color='black', dash='dash'),
          name = "true probability"
      ))
  
  # get correct and incorrect choices
  correct_choices = np.array([choice1[i] if (opt1Rewarded[i] and choice1[i]) or  (not opt1Rewarded[i] and not choice1[i]) else np.NAN for i in range(nTrials)])
  incorrect_choices = np.array([choice1[i] if (opt1Rewarded[i] and not choice1[i]) or  (not opt1Rewarded[i] and choice1[i]) else np.NAN for i in range(nTrials)])
  
  # plot choices as a scatterplot
  fig.add_trace(
      go.Scatter(
          x = list(range(nTrials)),
          y = correct_choices,
          mode = 'markers',
          marker = dict(size = 5, opacity=0.8, color="green"),
          name = "correct choice"
      ))
  
  fig.add_trace(
      go.Scatter(
          x = list(range(nTrials)),
          y = incorrect_choices,
          mode = 'markers',
          marker = dict(size = 5, opacity=0.8, color="red"),
          name = "incorrect choice"
      ))

  # label the axes
  fig.update_layout(xaxis_title="", yaxis_title="option 1 rewarded?")

  # plot reward magnitudes as lines
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
  
  # label the axes
  fig.update_layout(xaxis2_title="trial number", yaxis2_title="reward magnitude")

  # set x-axis range for all subplots
  # this sets the range for all x-axes
  fig.update_xaxes(range=[0, nTrials])
  # this is needed to set the range for the second subplot 
  fig.update_xaxes(range=[0, nTrials], row=2, col=1)

  return fig

def visualise_alpha_distributions(stableAlphas, volatileAlphas, title):
  fig = px.histogram(pd.DataFrame({"stable block": stableAlphas,
                                  "volatile block": volatileAlphas}).melt(),
                                  color="variable", x="value", marginal="box", barmode="overlay")

  fig.update_layout(title= title, xaxis_title="learning rate", legend_title_text="")
  fig.update_xaxes(range=[0, 1])
  
  fig.update_traces(xbins=dict(
        start=0.0,
        end=1.0,
        size=0.05
    ), 
    row = 1
    )

  fig.show()
  
  
def visualise_alpha_difference(stableAlphas, volatileAlphas, title):
  fig = px.histogram(pd.DataFrame({"volatile alpha - stable alpha": volatileAlphas - stableAlphas}).melt(),
                                  color="variable", x="value", marginal="box", barmode="overlay")

  fig.update_layout(title= title, xaxis_title="volatile alpha - stable alpha", legend_title_text="", showlegend=False)
  fig.update_xaxes(range=[-1, 1])
  
  fig.update_traces(xbins=dict(
        start=-1.0,
        end=1.0,
        size=0.1
    ), 
    row = 1
    )

  fig.show()
  
def plot_parameter_corrs(df):
  if any(df.columns == 'alphaStable'):
    if any(df.columns == 'omega'):
      fig = go.Figure(data=go.Splom(
                    dimensions=[dict(label='alpha stable',
                                    values=df.alphaStable),
                                dict(label='alpha volatile',
                                    values=df.alphaVolatile),
                                dict(label='beta',
                                    values=df.beta),
                                dict(label='omega',
                                    values=df['omega'])],
                    showupperhalf=False,
                    diagonal_visible=False, 
                    hovertemplate = 'participant ID: %{customdata}',
                    customdata = df.ID
                    ))
    else:
      fig = go.Figure(data=go.Splom(
                    dimensions=[dict(label='alpha stable',
                                    values=df.alphaStable),
                                dict(label='alpha volatile',
                                    values=df.alphaVolatile),
                                dict(label='beta',
                                    values=df.beta)],
                    showupperhalf=False,
                    diagonal_visible=False, 
                    hovertemplate = 'participant ID: %{customdata}',
                    customdata = df.ID
                    ))
  else:
    if any(df.columns == 'omega'):
      fig = go.Figure(data=go.Splom(
                    dimensions=[dict(label='alpha',
                                    values=df.alpha),
                                dict(label='beta',
                                    values=df.beta),
                                dict(label='omega',
                                    values=df['omega'])],
                    showupperhalf=False,
                    diagonal_visible=False, 
                    hovertemplate = 'participant ID: %{customdata}',
                    customdata = df.ID
                    ))
    else:
      fig = go.Figure(data=go.Splom(
                    dimensions=[dict(label='alpha',
                                    values=df.alphaStable),
                                dict(label='beta',
                                    values=df.beta)],
                    showupperhalf=False,
                    diagonal_visible=False, 
                    hovertemplate = 'participant ID: %{customdata}',
                    customdata = df.ID
                    ))
    
  fig.show()
