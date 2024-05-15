'''
Cell migration model
'''

import numpy as np
import pandas as pd

from tyssue import config, Sheet, PlanarGeometry
from tyssue.solvers import QSSolver

from tyssue.solvers.viscous import EulerSolver
from tyssue.draw import sheet_view

from tyssue.dynamics.factory import model_factory
from tyssue.dynamics import effectors, units

from tyssue.utils import to_nd
from tyssue.utils.testing import effector_tester, model_tester
from tyssue.behaviors import EventManager
from tyssue.behaviors.sheet import basic_events

from tyssue.draw import highlight_faces, create_gif


# =============================================================================
# Create a 2D patch of cells
# =============================================================================

geom = PlanarGeometry

sheet = Sheet.planar_sheet_2d('jam', 15, 15, 1, 1, noise=0.2)
geom.update_all(sheet)

sheet.remove(sheet.cut_out([[0, 8], [0, 8]]), trim_borders=True)
sheet.sanitize()
geom.scale(sheet, sheet.face_df.area.mean()**-0.5, ['x', 'y'])
geom.update_all(sheet)
sheet.reset_index()
sheet.reset_topo()
fig, ax = sheet_view(sheet, mode="quick")


# =============================================================================
# Define the relevant mechanical components
# =============================================================================

# Adding some gigling
class BrownianMotion(effectors.AbstractEffector):
    
    label = 'Brownian Motion'
    element = 'vert'
    specs = {"settings": {"temperature": 1e-3}}
    
    def energy(eptm):
        T = eptm.settings['temperature']
        return np.ones(eptm.Nv) * T / eptm.Nv
    
    def gradient(eptm):
        T = eptm.settings['temperature']
        scale = T/eptm.edge_df.length.mean()
        
        grad = pd.DataFrame(
            data=np.random.normal(0, scale=scale, size=(eptm.Nv, eptm.dim)),
            index=eptm.vert_df.index,
            columns=['g'+c for c in eptm.coords]
        )
        return grad, None


# =============================================================================
# Quasistatic gradient descent
# =============================================================================

model = model_factory(
    [effectors.PerimeterElasticity,
     effectors.FaceAreaElasticity])


model_specs = {
    'face': {
        'area_elasticity': 1.,
        'prefered_area': 1.,
        'perimeter_elasticity': 0.1, # 1/2r in the master equation
        'prefered_perimeter': 3.81,
        },
    'edge': {
        'ux': 0,
        'uy': 0,
        'is_active': 1,
        'line_tension': 0.0
    },
    'settings': {'temperature': 1e-2}
}
    
sheet.update_specs(model_specs, reset=True)

res = QSSolver().find_energy_min(sheet, PlanarGeometry, model)

print(res.message)

bck = sheet.copy()  #Back up so we can play with parameters.


# =============================================================================
# Pulling on a face
# =============================================================================

def tract(eptm, face, pull_axis, value, pull_column='line_tension', face_id=None):
    """Modeling face traction as shrinking apical junctions on the neighouring cells,
       As if pseudopods where pulling between the cells
    
    """
    
    pull_axis = np.asarray(pull_axis)
    edges = eptm.edge_df.query(f'face == {face}')
    verts = edges['srce'].values
    r_ai = edges[["r"+c for c in eptm.coords]].values
    proj = (r_ai * pull_axis[None, :]).sum(axis=1)
    
    pulling_vert = verts[np.argmax(proj)]
    
    v_edges = eptm.edge_df.query(
        f'(srce == {pulling_vert}) & (face != {face})'
    )
    pulling_edges = v_edges[~v_edges['trgt'].isin(verts)].index
    eptm.edge_df[pull_column] = 0

    if pulling_edges.size:
        eptm.edge_df.loc[pulling_edges, pull_column] = value
        
        
default_traction_spec = {
    "face": -1,
    "face_id": -1,
    "pull_axis": [0.1, 0.9],
    "value": 4,
    "pull_column": "line_tension"
}


from tyssue.utils.decorators import face_lookup

@face_lookup
def traction(sheet, manager, **kwargs):
    
    traction_spec = default_traction_spec
    traction_spec.update(**kwargs)
    pulling = tract(sheet, **traction_spec)
    
    manager.append(traction, **traction_spec)

pulled = 2


# =============================================================================
# Simple visualisation
# =============================================================================

highlight_faces(sheet.face_df, [pulled,], reset_visible=True)
fig, ax = sheet_view(
    sheet, 
    mode="2D", 
    face={"visible": True},
    edge={"head_width": 0.0}, 
    vert={"visible": False})
fig.set_size_inches(120, 120)



















'''
This is the end of the file.
'''
