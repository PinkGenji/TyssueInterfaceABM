"""
This script shows the behaviour function of T2 and how it runs with Euler solver
"""

# Load all required modules.
import random
import numpy as np
import os
import sys
import re
import matplotlib.pyplot as plt
from tyssue import Sheet, History, PlanarGeometry
from tyssue.topology.sheet_topology import remove_face, type1_transition
from tyssue.dynamics import effectors, model_factory
from tyssue.behaviors import EventManager
from tyssue.solvers.viscous import EulerSolver

# Plotting related
from tyssue.draw import sheet_view
from tyssue.config.draw import sheet_spec
import imageio.v2 as imageio

# Set relative path, then import my own functions.
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Model with multiple cell class'))
sys.path.append(model_path)
print("Model path:", model_path)
print("Files in directory:", os.listdir(model_path))
import my_headers as mh
import T3_function as T3


# Functions used in this script.

def extract_PNGnumber(fname):
    """
    Helper function to extract the numeric part from a filename for later use.
    For example, from "frame_12.png", it extracts 12
    """
    match = re.search(r'\d+', fname)
    return int(match.group()) if match else -1  # If no number found, use -1

def drop_face(sheet, face, **kwargs):
    """
    Removes the face indexed by "face" and all associated edges
    """
    edge = sheet.edge_df.loc[(sheet.edge_df['face'] == face)].index
    print(f"Dropping face '{face}'")
    sheet.remove(edge, **kwargs)

random.seed(42)  # Controls Python's random module (e.g. event shuffling)
np.random.seed(42)  # Controls NumPy's RNG (e.g. vertex positions, topology)

rng = np.random.default_rng(70)  # Seed the random number generator for my own division function.

# Generate the initial cell sheet for bilayer.
geom = PlanarGeometry
print('\n Now we change the initial geometry to bilayer.')
num_x = 16
num_y = 4

sheet = Sheet.planar_sheet_2d(identifier='bilayer', nx=num_x, ny=num_y, distx=1, disty=1)
geom.update_all(sheet)
# Updates the sheet geometry by updating: * the edge vector coordinates * the edge lengths * the face centroids
# * the normals to each edge associated face * the face areas.

# remove non-enclosed faces
sheet.remove(sheet.get_invalid())

# Repeatedly remove all non-hexagonal faces until none remain
while np.any(sheet.face_df['num_sides'].values != 6):
    bad_face = sheet.face_df[sheet.face_df['num_sides'] != 6].index[0]
    drop_face(sheet, bad_face)
# Update the geometry and computes the grabs all pairs of half-edges.
geom.update_all(sheet)
sheet.get_opposite()

# Plot the figure to see the initial setup is what we want.
fig, ax = sheet_view(sheet)
ax.set_title("Initial Bilayer Setup")  # Adding title
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face, fontsize=10, color="r")
plt.show()
print('Initial geometry plot generated. \n')


# Load the effectors, then explicitly define what parameters are using in the simulation.
model = model_factory([
    effectors.LineTension,
    effectors.FaceContractility,
    effectors.FaceAreaElasticity
])
sheet.vert_df['viscosity'] = 1.0
specs = {
    'edge': {
        'is_active': 1,
        'line_tension': 10,
        'ux': 0.0,
        'uy': 0.0,
        'uz': 0.0
    },
    'face': {
        'area_elasticity': 200,
        'contractility': 5,
        'is_alive': 1,
        'prefered_area': 0.8},
    'settings': {
        'grad_norm_factor': 1.0,
        'nrj_norm_factor': 1.0
    },
    'vert': {
            'is_active': 1
    }
}
# Update the specs (adds / changes the values in the dataframes' columns)
sheet.update_specs(specs, reset=True)
sheet.get_opposite()
geom.update_all(sheet)

# Adjust coefficient for cell-boundary adhesion force.
sheet.edge_df.loc[sheet.edge_df["opposite"] == -1, "line_tension"] *= 2
geom.update_all(sheet)
print('The parameters used for force calculation is added to the model. \n')

# Set the value of constants for mesh restructure, which are parts of my own solver in loop.
t2_threshold = sheet.face_df['area'].mean()/10
manager = EventManager('face')


# Run the Euler solver
solver = EulerSolver(sheet, geom, model, manager=manager)
solver.solve(tf=5, dt=0.001)
geom.update_all(sheet)
fig, ax = sheet_view(sheet)
plt.show()





print('\n This is the end of this script. (＾• ω •＾) ')
