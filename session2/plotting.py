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
    magOpt1     = None,
    magOpt2     = None,
    probOpt1    = None,
    choiceProb1 = None,
    choice1     = None
    ):
  '''
  Plots the experimental schedule and the RL model estimate using plotly.

    Parameters:
        opt1rewarded(bool array): True if option 1 is rewarded on a trial, False
          if option 2 is rewarded on a trial.
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

    # label the axes
    fig.update_layout(xaxis2_title="trial number", yaxis2_title="reward magnitude")

  # set x-axis range for all subplots
  # this sets the range for all x-axes
  fig.update_xaxes(range=[0, nTrials])
  # this is needed to set the range for the second subplot if it exists
  if magOpt1 is not None:
    fig.update_xaxes(range=[0, nTrials], row=2, col=1)

  return fig

def visualise_utility_function(utility_function, omega = False, nSamples = 100):
  '''
  Visualises a utility function using plotly.

    Parameters:
        utility_function(function): The utility function to visualise.
        omega(bool): Whether the utility function has an omega parameter.
        nSamples(int): The number of samples to take from the utility function
          to visualise it.
  '''

  # get a slider for omega
  omegaSlider = widgets.FloatSlider(
                            value=0.5,
                            max=1,
                            min=0,
                            step=0.01,
                            description='omega:',
                            continuous_update=False)

  # the range of mag and prob values for which we want to visualise the function
  magRange  = np.linspace(1, 100, nSamples)
  probRange = np.linspace(0, 1, nSamples)

  # function to fill in the utility matrix
  def compute_utilityMatrix(
      utility_function = utility_function, 
      show_omega = omega, 
      nSamples = nSamples, 
      omega = omegaSlider.value
      ):

    # empty matrix to fill in with utility values
    utilityMatrix = np.zeros((nSamples,nSamples))

    # loop through all possible values of mag and prob and get their utility
    for mag in range(nSamples):
      for prob in range(nSamples):
        if show_omega:
          utilityMatrix[mag,prob] = utility_function(magRange[mag], probRange[prob], omega)
        else:
          utilityMatrix[mag,prob] = utility_function(magRange[mag], probRange[prob])
    
    return utilityMatrix
  
  utilityMatrix = compute_utilityMatrix(utility_function)

  # make a 3d plot of the utility function
  fig = go.Figure(go.Surface(z = utilityMatrix,
                            y = magRange,
                            x = probRange))

  fig.update_scenes(yaxis_title='magnitude',
                    xaxis_title='probability',
                    zaxis_title='utility')

  fig.update_layout(scene_camera=dict(eye=dict(x=-2, y=-2, z=2)))

  # if omega is true, add a slider to change omega
  if omega:
    # make the figure interactive
    fig = go.FigureWidget(fig)

    # function that triggers when the omega value changes
    def change_omega(change):
      # calculate the utility matrix for the new omega value
      utilityMatrix = compute_utilityMatrix(omega = omegaSlider.value)
      # update the figure
      with fig.batch_update():
        fig.data[0].z = utilityMatrix
    
    # listen to changes of the omega slider
    omegaSlider.observe(change_omega, names="value")

    # show the slider and figure
    display(widgets.VBox([omegaSlider, fig]))

  else:
     display(fig)

 
def visualise_softmax(softmax):
  '''
  Visualises a softmax function using plotly.
  
      Parameters:
          softmax(function): The softmax function to visualise.
    '''

  # define a slider for the inverse temperature
  betaSlider = widgets.FloatSlider(
                              value=3,
                              max=50,
                              min=0,
                              step=0.01,
                              description='beta:',
                              continuous_update=False)

  # the initial beta to display
  beta = 3
  utilityRange = np.linspace(-2, 2, 1000)

  # start a new figure
  fig = go.FigureWidget(go.Scatter(
            x = utilityRange,
            y = softmax(utilityRange, 0, beta),
            mode = 'lines',
            line = dict(color='black')
        ))


  # label the axes
  fig.update_layout(xaxis_title="utility 1 - utility 2", yaxis_title="probability of choosing option 1")

  # set the range for the y-axis
  fig.update_yaxes(range=[0, 1])

  # function that triggers when the beta value changes
  def change_beta(change):
    # get the current value of beta
    beta = betaSlider.value
    # calculate the corresponding utilities
    u = softmax(utilityRange, 0, beta)
    # update the figure
    with fig.batch_update():
      fig.data[0].y = u

  # listen to changes of the beta slider
  betaSlider.observe(change_beta, names="value")

  # show the slider and figure
  display(widgets.VBox([betaSlider, fig]))