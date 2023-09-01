# numpy is a libarary used to do all kinds of mathematical operations
import numpy as np

# we use plotly to make our figures
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px

# set the style of the plotly figures
pio.templates.default = "none"

# this allows us to make interactive figures
import ipywidgets as widgets

def plot_schedule(
    opt1Rewarded, 
    trueProbability, 
    probOpt1 = None
    ):
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

def plot_interactive_RL_model(
    opt1Rewarded, 
    trueProbability, 
    RL_model,
    generate_schedule, 
    change_trueProb = True,
    ):
  '''
  Plots the experimental schedule and the RL model estimate with sliders to
  change the model parameters.

    Parameters:
        opt1rewarded(int array): 1 if option 1 is rewarded on a trial, 0 if
          option 2 is rewarded on a trial.
        trueProbability(float array): The probability with which option 1 is
          rewareded on each trial.
        RL_model(function): the RL model function to use to generate probOpt1.
        change_trueProb(bool): whether to include a slider to change the true
          probability. Defaults to True.
  '''

  # compute number of trials
  nTrials = len(opt1Rewarded)

  # this initiates some sliders that can be used to dynamically change aprameter values
  alphaSlider        = widgets.FloatSlider(
                                  value=0.1,
                                  max=1,
                                  min=0.001,
                                  step=0.01,
                                  description='alpha:',
                                  continuous_update=False
                                  )

  startingProbSlider = widgets.FloatSlider(
                                  value=0.5,
                                  max=1,
                                  min=0,
                                  step =0.01,
                                  description='startingProb:',
                                  continuous_update=False
                                  )
  
  trueProbSlider     = widgets.FloatSlider(
                                  value=0.8,
                                  max=1,
                                  description='trueProb:',
                                  continuous_update=False
                                  )

  sliders = widgets.VBox(children=[alphaSlider,
                                  startingProbSlider,
                                  trueProbSlider])
  
  probOpt1 = RL_model(opt1Rewarded,alphaSlider.value,startingProbSlider.value)

  # call the figure function we wrote and make it interactive
  fig = go.FigureWidget(plot_schedule(opt1Rewarded,trueProbability,probOpt1))

  # function to run if alpha or startingProb have changed
  def change_model(change):
    # get current values
    opt1Rewarded = fig.data[0].y
    trueProbability = trueProbSlider.value
    # rerun the RL model
    probOpt1 = RL_model(opt1Rewarded,alphaSlider.value,startingProbSlider.value)
    # update the figure
    with fig.batch_update():
      fig.data[2].y = probOpt1

  # function to run if trueProbability has changed
  def change_schedule(change):
    # generate a new schedule
    trueProbability = np.ones(nTrials, dtype = float)*trueProbSlider.value
    opt1Rewarded = generate_schedule(trueProbability)
    # rerun the RL model
    probOpt1 = RL_model(opt1Rewarded,alphaSlider.value,startingProbSlider.value)
    # update the figure
    with fig.batch_update():
      fig.data[0].y = opt1Rewarded
      fig.data[1].y = trueProbability
      fig.data[2].y = probOpt1

  # run the functions if a slider value changes
  alphaSlider.observe(change_model, names="value")
  startingProbSlider.observe(change_model, names="value")
  if change_trueProb:
    trueProbSlider.observe(change_schedule, names="value")

  # show the figure and the sliders
  display(widgets.VBox([sliders, fig]))