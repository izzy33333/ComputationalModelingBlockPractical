# we use plotly to make our figures
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px
pio.templates.default = "none"

# this allows us to make interactive figures
import ipywidgets as widgets

# get the modeling scripts we wrote
from . import modeling

def plot_schedule(
    *,
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
        
        opt1rewarded(bool array): True if option 1 is rewarded on a trial, False if option 2 is rewarded on a trial.
        
        trueProbability(float array): The probability with which option 1 is rewareded on each trial.
        
        magOpt1(int array): The reward magnitude of option 1. Defaults to None, which excludes it from the plot.
        
        magOpt2(int array): The reward magnitude of option 1. Defaults to None, which excludes it from the plot.
        
        probOpt1(float array): how likely option 1 is rewarded on each trial according to the RL model. Defaults to None, which excludes it from the plot.
        
        choiceProb1(float array): the probability of choosing option 1 on each trial according to the RL model. Defaults to None, which excludes it from the plot.
        
        choice1(int array): whether option 1 (1) or option 2 (2) was chosen on each. Defaults to None, which excludes it from the plot.
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


def plot_interative_schedule(
      opt1Rewarded,
      trueProbability
    ):
    # this initiates some sliders that can be used to dynamically change parameter values
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

    # call the figure function we wrote and make it interactive
    fig = go.FigureWidget(plot_schedule(opt1Rewarded,trueProbability,probOpt1))

    # function to run if alpha or startingProb have changed
    def change_model(change):
        # get current values
        opt1Rewarded = fig.data[0].y
        trueProbability = trueProbSlider.value
        # rerun the RL model
        probOpt1 = modeling.simulate_RL_model(
           opt1Rewarded,
           None,
           None,
           alphaSlider.value,
           None,
           startingProb = startingProbSlider.value
           )
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
    trueProbSlider.observe(change_schedule, names="value")

    # show the figure and the sliders
    display(widgets.VBox([sliders, fig]))
