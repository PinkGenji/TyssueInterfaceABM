# -*- coding: utf-8 -*-
"""
This file is for learning 04Solvers of the tyssue package.
"""

# ignore FutureWarning from pandas.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display


from tyssue import config, Sheet, SheetGeometry, History, EventManager
from tyssue.draw import sheet_view
from tyssue.generation import three_faces_sheet
from tyssue.draw.plt_draw import plot_forces

from tyssue.dynamics import PlanarModel

from tyssue.solvers.viscous import EulerSolver
from tyssue.draw.plt_draw import create_gif


geom  = SheetGeometry
model = PlanarModel

sheet = Sheet.planar_sheet_3d('planar', nx=6, ny=6, 
                             distx=1, disty=1)
sheet.sanitize(trim_borders=True, order_edges=True)
geom.update_all(sheet)
fig, ax = sheet_view(sheet)


'''
The history object

The HIstory class defines the object in charge of stroing the evolving epithelium
during the course of the simulation. It allows to access to different time points
of a simulation from 1 unique epithelium.

Most of the time, we use HistoryHDF5 class, that writes each time step to a file,
which can be useful for big files. It is also possible to read an hf5 file to analyze
a simulation later.

In the solver, we use the history.record method to store the epithelium.
In the create_grif function, we use the history.retrieve method to get back the epithelium
at a given time point.

'''

history = History(sheet, save_every=2, dt=1)

for i in range(10):
    
    geom.scale(sheet, 1.02, list('xy'))
    geom.update_all(sheet)
    # record only every `save_every` time 
    history.record()

create_gif(history, 'simple_growth.gif', num_frames=len(history))


Image('simple_growth.gif')

# retrieve function returns an epithelium of the same type as the original.
type(history.retrieve(5))

# Iterate over an history object yields a time and a sheet object:
for t, sheet in history:
    print(f"mean area at {t}: {sheet.face_df.area.mean():.3f}", )

# The vert_h, edge_h and face_h Dataframes hold the history:
history.vert_h.head()

'''
We can plot the evolution of a column over time!!
'''
fig, ax = plt.subplots()

ax.scatter(history.face_h['time'], history.face_h['area'], alpha=0.2, s=12)

history.face_h.groupby('time').area.mean().plot(ax=ax)


'''
Quasistatic solver

A common way to describe an epithelium is with the quasistatic approximation:
we assume that at any given point in time, without exterior pertubation,
the epithelium is at an energy minimum. For a given expression of the model's
hamiltonian, we thus seasrch the position of the vertices corresponding to the 
minimum energy.


'''

# Following is for the Farhadifar model. The energy is the sum of an area elasticity, 
# a contractility and a line tension:

from tyssue.config.dynamics import quasistatic_plane_spec
from tyssue.dynamics.planar_vertex_model import PlanarModel as smodel
from tyssue.solvers import QSSolver
from pprint import pprint

specs = {
    'edge': {
        'is_active': 1,
        'line_tension': 0.12,
        'ux': 0.0,
        'uy': 0.0,
        'uz': 0.0
    },
   'face': {
       'area_elasticity': 1.0,
       'contractility': 0.04,
       'is_alive': 1,
       'prefered_area': 1.0},
   'settings': {
       'grad_norm_factor': 1.0,
       'nrj_norm_factor': 1.0
   },
   'vert': {
       'is_active': 1
   }
}


# Update the specs (adds / changes the values in the dataframes' columns)
sheet.update_specs(specs)

pprint(specs)

E_t = smodel.compute_energy(sheet)
print('Total energy: ', E_t)

smodel.compute_gradient(sheet).head()

'''
The energy minimum is found with a gradient descent strategy, the vertices are
displaced in the direction opposite to the spatial derivative of the energy.
Actually, this defines the force on the vertices.

The gradient descent algorithm is provided by scipy minimize function, with the
L-BFGSB method by default.
'''

# Find energy minimum
solver = QSSolver()
res = solver.find_energy_min(sheet, geom, smodel)

print("Successfull gradient descent? ", res['success'])
fig, ax = sheet_view(sheet)

fig.set_size_inches(10, 10)
ax.set_aspect('equal')


# Keyword arguments to find_energy_min method are passed to scipy's minimize,
# so it is possible for example to reduce the termination criteria.

sheet.face_df.loc[4, 'prefered_area'] = 2.0
res = solver.find_energy_min(sheet, geom, smodel, options = {'ftol': 1e-2})

print('Boolean for the convergence success: ')
print(res.success)

print('Number of function evaluations: ')

print(res.nfev)

print('Information on the convergence: ')
print(res.message)


'''
The solver objecct provides facilities to approximate the gradient and evaluate
the error between the actual gradient and the approximate one.
'''

print('Total gradient error: ')
solver.check_grad(sheet, geom, model)

app_grad = solver.approx_grad(sheet, geom, model)
app_grad = pd.DataFrame(app_grad.reshape((-1, 3)), columns=['gx', 'gy', 'gz'])
app_grad.head()

'''
Simple forward Euler solver:

The Eulersolver is time dependent, the model is used in the same way as the
quasistatic solver.

'''

sheet = Sheet('3', *three_faces_sheet())
geom.update_all(sheet)
sheet.settings['threshold_length'] = 1e-3

sheet.update_specs(config.dynamics.quasistatic_plane_spec())
sheet.face_df["prefered_area"] = sheet.face_df["area"].mean()
history = History(sheet) #, extra_cols={"edge":["dx", "dy", "sx", "sy", "tx", "ty"]})

sheet.vert_df['viscosity'] = 1.0
sheet.edge_df.loc[[0, 17],  'line_tension'] *= 4
sheet.face_df.loc[1,  'prefered_area'] *= 1.2

fig, ax = plot_forces(sheet, geom, model, ['x', 'y'], 1)


# Solver instanciation: contrary to the quasistatic solver, this sovler needs 
# the sheet, goemetry and model at instanciation time.
solver = EulerSolver(
    sheet,
    geom,
    model,
    history=history,
    auto_reconnect=True)

'''
The solver's solve method accepts a on_topo_change function as argument.
This function is executed each time a topology change occurs.
Here, we reset the line tension to its original value.

'''
def on_topo_change(sheet):
    print('Topology changed!\n')
    print("reseting tension")
    sheet.edge_df["line_tension"] = sheet.specs["edge"]["line_tension"]

# Solving from t = 0 to t = 15.

res = solver.solve(tf=15, dt=0.05, on_topo_change=on_topo_change,
                   topo_change_args=(solver.eptm,))

# Showing the results

create_gif(solver.history, "sheet3.gif", num_frames=120)

Image("sheet3.gif")

# The Image seems not working. Cannot get a gif picture.






'''
This is the end of the file. :)
'''
