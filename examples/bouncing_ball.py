
# %%
import numpy as np
import pandas as pd

tmax = 5
num = 1000


def analytical_solution(N_hits, g=9.81, f=0.05, x0=1):
    t_hits = []
    summation = 0
    for i in range(N_hits):
        summation += (2 * (1 - f) ** (i))
        t_hit = np.sqrt(2 * x0 / g) * (summation - 1)
        t_hits.append(t_hit)

    t_hits = np.array(t_hits)
    #    t_hits.shape = (1,len(t_hits))
    return t_hits



# %%

t_hits = analytical_solution(N_hits=10)
pd.DataFrame(t_hits, columns=["Total time"])



import sys
import os
from numerous.engine.system import Item
from numerous.engine.model import Model
from numerous.engine.system import Subsystem
from numerous.engine.simulation import Simulation
from numerous.engine.system import ConnectorTwoWay
from numerous.engine import VariableType, VariableDescription, OverloadAction
from numerous.engine.simulation.solvers.base_solver import SolverType

from numerous.multiphysics import EquationBase, Equation

from numerous.utils.historian import LocalHistorian as HistoryDataFrame
from datetime import timedelta
import plotly.graph_objs as go


# %%


class Link_x():
    def __init__(self, handles=[]):
        self.handles = handles
        for f in handles:
            f.layout.on_change(self.callback, 'xaxis.range')

    def callback(self, layout, xrange):
        for handle in self.handles:
            handle.layout.xaxis['range'] = xrange


class Plotfig():
    def __init__(self):
        self.list = []

    def link_x(self):
        for f in self.list:
            f.layout.on_change(self.link_x_callback, 'xaxis.range')

    def link_x_callback(self, layout, xrange):
        for f in self.list:
            f.layout.xaxis['range'] = xrange

    def add(self, res, y, xlabel=None, ylabel=None, avr=[False, ], xlim=[None, None], ylim=[None, None]):

        legend_x = 1
        legend_y = 0.5

        data = []
        for yi in y:

            data.append(go.Scatter(x=res[yi]['time'], y=res[yi]['values'], name=yi, mode='lines'))
            if avr[0] == True:
                N = avr[1]
                y_padded = np.pad(res[yi]['values'], (N // 2, N - 1 - N // 2), mode='edge')
                yavr = np.convolve(y_padded, np.ones((N,)) / N, mode='valid')
                data.append(go.Scatter(x=res[yi]['time'], y=yavr, name=yi + '_mean_' + str(N)))

        if (xlim[0] == None) and (xlim[1] == None):
            xautorange = True
        else:
            xautorange = False

        if (ylim[0] == None) and (ylim[1] == None):
            yautorange = True
        else:
            yautorange = False

        layout = go.Layout(
            xaxis=dict(
                autorange=xautorange,
                range=xlim,
                showgrid=True,
                zeroline=False,
                showline=False,
                title=xlabel,
                showticklabels=True
            ),
            yaxis=dict(
                autorange=yautorange,
                range=ylim,
                showgrid=True,
                zeroline=False,
                showline=False,
                title=ylabel,
                showticklabels=True
            ),
            showlegend=True,
        )

        f = go.FigureWidget(data=data, layout=layout)

        self.list.append(f)
        return f

    def show(self):
        for f in self.list:
            f.show()


class Collection():
    def __init__(self):
        self.res = {}

    def alias(self, historydataframe, y={}):

        for key in list(y.keys()):
            self.res.update({key: {'values': [], 'time': []}})

        for index, row in historydataframe.iterrows():
            for tag in y:
                self.res[tag]['values'].append(row[y[tag]])
                self.res[tag]['time'].append(index)




class Bouncing(EquationBase):
    def __init__(self, g=9.81, f_loss=0.05, x=1, v=0):
        super().__init__(tag='bouncing_eq')
        self.add_constant('g', g)
        self.add_constant('f_loss', f_loss)
        self.add_parameter('t_hit', 0)
        self.add_state('x', x)
        self.add_state('v', v)

    @Equation()
    def eval(self, scope):
        scope.x_dot = scope.v  # Position
        scope.v_dot = -scope.g  # Velocity



# %%

class Ball(Item):
    def __init__(self, tag="ball", g=9.81, f_loss=5, x0=1, v0=0):
        super().__init__(tag)

        t1 = self.create_namespace('t1')
        t1.add_equations([Bouncing(g=g, f_loss=f_loss, x=x0, v=v0)])



# %%

def ms1(simple_item):
    class S1(Subsystem):
        def __init__(self, tag):
            super().__init__(tag)
            self.register_items([simple_item])

    return S1('S1')


# %%

model_system_1 = ms1(Ball(tag="ball", g=9.81, f_loss=0.05))




# %%

# returns position to find zero crossing using root finding algorithm of scipy solver
def hitground_event_fun(t, states):
    return states['S1.ball.t1.x']






# change direction of movement upon event detection and reduce velocity
def hitground_event_callback_fun(t, variables):
    velocity = variables['S1.ball.t1.v']
    velocity = -velocity * (1 - variables['S1.ball.t1.f_loss'])
    variables['S1.ball.t1.v'] = velocity
    variables['S1.ball.t1.t_hit'] = t


model_system_2 = ms1(Ball(tag="ball", g=9.81, f_loss=0.05))


# %%


def solve_model_system_event_fun(model_system):
    m1 = Model(model_system,use_llvm=True)

    m1.add_event("hitground_event", hitground_event_fun, hitground_event_callback_fun)

    sim = Simulation(m1, t_start=0, t_stop=tmax, num=num, solver_type=SolverType.NUMEROUS)

    sol = sim.solve()
    return sim, sol


# s1,sol1  = solve_model_system_fun(model_system_1)


# %%

sim2, sol2 = solve_model_system_event_fun(model_system_2)

# %%

collections = Collection()
collections.alias(sim2.model.historian_df, y={'velocity_event': sim2.model.path_to_variable['S1.ball.t1.v'].id,
                                              'position_event': sim2.model.path_to_variable['S1.ball.t1.x'].id})
f = Plotfig()
f.add(collections.res, y=['velocity_event'], xlabel='Time (s)', ylabel='Velocity (m/s)')
f.add(collections.res, y=['position_event'], xlabel='Time (s)', ylabel='Position (m)')
f.link_x()
f.show()

# %%
