# -*- coding: utf-8 -*-
"""
3d cell division
"""
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

from tyssue.io.hdf5 import save_datasets, load_datasets
# redefine cell_division from monolayer related topology module
from tyssue.topology.monolayer_topology import cell_division

from tyssue.core.monolayer import Monolayer
from tyssue.geometry.bulk_geometry import ClosedMonolayerGeometry as monolayer_geom
from tyssue.dynamics.bulk_model import ClosedMonolayerModel
from tyssue.draw import highlight_cells

datasets = load_datasets('small_ellipsoid.hf5',data_names=['vert', 'edge','face', 'cell'])

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
solver = QSSolver()
res = solver.find_energy_min(monolayer, monolayer_geom, ClosedMonolayerModel)
mother = 8
daughter = cell_division(monolayer, mother, orientation='vertical')

monolayer.validate()
draw_specs = config.draw.sheet_spec()

rho = np.linalg.norm(monolayer.vert_df[monolayer.coords], axis=1)
draw_specs['edge']['color'] = rho
draw_specs['face']['visible'] = True

ipv.clear()
highlight_cells(monolayer, mother, reset_visible=True)
fig, mesh = sheet_view(monolayer, mode="3D", coords=['z', 'x', 'y'], **draw_specs)
fig

mother = 18
daughter = cell_division(monolayer, mother, orientation='horizontal')
monolayer.validate()

rho = np.linalg.norm(monolayer.vert_df[monolayer.coords], axis=1)
draw_specs['edge']['color'] = rho
draw_specs['face']['visible'] = True
ipv.clear()
highlight_cells(monolayer, mother)
fig, mesh = sheet_view(monolayer, mode="3D", coords=['z', 'x', 'y'], **draw_specs)
fig


