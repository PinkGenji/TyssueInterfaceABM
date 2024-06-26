# -*- coding: utf-8 -*-
"""
This file is for learning 06CellDivision of the tyssue package.
"""

import os
import pandas as pd
import numpy as np
import json
import matplotlib.pylab as plt
import ipyvolume as ipv


from tyssue import Sheet, config


from tyssue.geometry.planar_geometry import PlanarGeometry as geom
from tyssue.solvers.quasistatic import QSSolver
from tyssue.dynamics.planar_vertex_model import PlanarModel as model

from tyssue.draw import sheet_view
from tyssue.stores import load_datasets

from tyssue.topology.sheet_topology import remove_face, cell_division

'''
On a 2D mesh:

'''

solver = QSSolver()
sheet = Sheet.planar_sheet_2d('division', 6, 6, 1, 1)
sheet.sanitize(trim_borders=True, order_edges=True)
geom.update_all(sheet)

sheet.get_opposite()

# ## Set up the model
nondim_specs = config.dynamics.quasistatic_plane_spec()
dim_model_specs = model.dimensionalize(nondim_specs)
sheet.update_specs(dim_model_specs, reset=True)

print("Number of cells: {}\n"
      "          edges: {}\n"
      "          vertices: {}\n".format(sheet.Nf, sheet.Ne, sheet.Nv))

# ## Minimize energy
res = solver.find_energy_min(sheet, geom, model)

# ## View the result
draw_specs = config.draw.sheet_spec()
draw_specs['vert']['visible'] = False
draw_specs['edge']['head_width'] = 0  # values other than 0 gives error.
fig, ax = sheet_view(sheet, **draw_specs)
fig.set_size_inches(12, 5)

# Generate the daughter cell.

daughter = cell_division(sheet, 7, geom, angle=np.pi/2)

res = solver.find_energy_min(sheet, geom, model)
print(res['success'])

fig, ax = sheet_view(sheet, **draw_specs)
fig.set_size_inches(12, 5)


from tyssue.io.hdf5 import save_datasets, load_datasets
# redefine cell_division from monolayer related topology module
from tyssue.topology.monolayer_topology import cell_division

from tyssue.core.monolayer import Monolayer
from tyssue.geometry.bulk_geometry import ClosedMonolayerGeometry as monolayer_geom
from tyssue.dynamics.bulk_model import ClosedMonolayerModel
from tyssue.draw import highlight_cells

wd = r"C:\Users\lyu195\Documents\GitHub\tyssueHello"
os.chdir(wd)

os.path.isfile('small_ellipsoid.hf5')  #Check working directory set correctly.

h5store = 'small_ellipsoid.hf5'

datasets = load_datasets(h5store, data_names=['vert', 'edge', 'face', 'cell'])

specs = config.geometry.bulk_spec()
monolayer = Monolayer('ell', datasets, specs)
monolayer_geom.update_all(monolayer)

specs = {
    "edge": {
        "line_tension": 0.0,
    },
    "face": {
        "contractility": 0.01,
    },
    "cell": {
        "prefered_vol": monolayer.cell_df['vol'].median(),
        "vol_elasticity": 0.1,
        "prefered_area": monolayer.cell_df['area'].median(),
        "area_elasticity": 0.1,
    },
    "settings": {
        'lumen_prefered_vol': monolayer.settings['lumen_vol'],
        'lumen_vol_elasticity': 1e-2

    }
}

monolayer.update_specs(specs, reset=True)

res = solver.find_energy_min(monolayer, monolayer_geom, ClosedMonolayerModel)

mother = 8
daughter = cell_division(monolayer, mother, 
                         orientation='vertical')

monolayer.validate()

# =============================================================================
#
# We shall skip the following 3D drawing, since tyssue is not very well written
# for 3D drawing.
#
# rho = np.linalg.norm(monolayer.vert_df[monolayer.coords], axis=1)
# draw_specs['edge']['color'] = rho
# draw_specs['face']['visible'] = True
# 
# ipv.clear()
# highlight_cells(monolayer, mother, reset_visible=True)
# fig, mesh = sheet_view(monolayer, mode="3D",coords=['z', 'x', 'y'], **draw_specs)
# fig
# =============================================================================



'''
This is the end of the file.
'''
