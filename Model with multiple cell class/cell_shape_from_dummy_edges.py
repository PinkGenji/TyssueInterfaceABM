"""
This script initialise a honey comb shape of cells. Different edges from the centre cell is set to be dummy edges.
"""

# Load all required modules.
import numpy as np
import re
import matplotlib.pyplot as plt
from tyssue import Sheet
from tyssue.topology.sheet_topology import remove_face, cell_division, fuse_single_cell, auto_dummy_edges
from tyssue.topology.base_topology import close_face, drop_face
from tyssue import PlanarGeometry as geom #for simple 2d geometry
from tyssue.dynamics import effectors, model_factory
from tyssue.solvers import QSSolver

# 2D plotting
from tyssue.draw import sheet_view
from tyssue.draw.plt_draw import plot_forces
from tyssue.topology.sheet_topology import cell_division, boundary_ids
from tyssue.topology.sheet_topology import T3_transition as T3
from tyssue.config.draw import sheet_spec

from src.tyssue import Epithelium


def update_draw_specs(sheet, draw_specs):
    """
    Update drawing specifications for faces and edges based on:
    - cell_class (STB, boundary_fixed, others)
    - is_active (dummy vs real edges)
    """
    # --- FACE COLORS ---
    # boundary_fixed = 0.1 (dark purple)
    # STB            = 0.5 (mid purple)
    # everything else = 0.9 (light background)
    face_color_map = {
        'boundary_fixed': 0.1,
        'STB': 0.5,
    }
    sheet.face_df['color'] = (
        sheet.face_df['cell_class']
        .map(face_color_map)
        .fillna(0.9)
    )
    draw_specs['face']['visible'] = True
    draw_specs['face']['color'] = sheet.face_df['color']
    draw_specs['face']['alpha'] = 0.2
    # --- EDGE WIDTHS ---
    # inactive (0) → thick (2)
    # active   (1) → thin (0.5)
    sheet.edge_df['width'] = sheet.edge_df['is_active'].map({0: 2, 1: 0.5})
    draw_specs['edge']['visible'] = True
    draw_specs['edge']['width'] = sheet.edge_df['width']
    return draw_specs

# Generate the initial cell sheet for bilayer.
print('\n Now we change the initial geometry to bilayer.')
num_x = 16
num_y = 4

sheet =Sheet.planar_sheet_2d(identifier='bilayer', nx = num_x, ny = num_y, distx = 1, disty = 1)
geom.update_all(sheet)
#Updates the sheet geometry by updating: * the edge vector coordinates * the edge lengths * the face centroids
# * the normals to each edge associated face * the face areas.
# remove non-enclosed faces
sheet.remove(sheet.get_invalid())

# Repeatedly remove all non-hexagonal faces until none remain
while np.any(sheet.face_df['num_sides'].values != 6):
    bad_face = sheet.face_df[sheet.face_df['num_sides'] != 6].index[0]
    drop_face(sheet, bad_face)

sheet.reset_index(order=True)   #continuous indices in all df, vertices clockwise
geom.update_all(sheet)
sheet.get_extra_indices()

# Plot the figure to see the initial setup is what we want.
fig, ax = sheet_view(sheet)
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)
ax.set_title("Initial Setup")
plt.tight_layout()
plt.show()

for i in sheet.face_df.index:
    if i in [19,20,21]:
        sheet.face_df.loc[i,"cell_class"] = "STB"
    else:
        sheet.face_df.loc[i,"cell_class"] = "CT"

"""
Assign dynamics to the model.
"""
# Add dynamics to the model, start with effectors, then change values.
model = model_factory([
    effectors.LineTension,
    effectors.PerimeterElasticity,
    effectors.FaceAreaElasticity
])

specs = {
    'edge': {
        'is_active': 1,
        'line_tension': 10,
        'ux': 0.0,
        'uy': 0.0,
        'uz': 0.0
    },
    'face': {
        'area_elasticity': 110,
        'perimeter_elasticity': 10,
        'is_alive': 1,
        'prefered_perimeter': 3.8,
        'prefered_area': 2},
    'settings': {
        'grad_norm_factor': 1.0,
        'nrj_norm_factor': 1.0
    },
    'vert': {
        'is_active': 1
    }
}
sheet.vert_df['viscosity'] = 1
# Update the specs (adds / changes the values in the dataframes' columns)
sheet.update_specs(specs, reset=True)
geom.update_all(sheet)

# Adjust for cell-boundary adhesion force.
for i in sheet.edge_df.index:
    if sheet.edge_df.loc[i, 'opposite'] == -1:
        sheet.edge_df.loc[i, 'line_tension'] *= 2
    else:
        continue
geom.update_all(sheet)

# Use QS solver to start with the steady state of the system.
solver = QSSolver()
res = solver.find_energy_min(sheet, geom, model)
print("Successfull gradient descent? ", res['success'])

# Fix the boundary
corner_cells = [0, num_x-3, num_x-2, len(sheet.face_df)-1]
fix_vert_set = set()
for cell in corner_cells:
    # Assign these cells into 'boundary' class, hence no cell class change on them.
    sheet.face_df.loc[cell,"cell_class"] = "boundary_fixed"
    sheet.face_df.loc[cell,"timer"] = np.nan
    vert_list = sheet.edge_df[sheet.edge_df['face'] == cell][['srce', 'trgt']].values.flatten()
    fix_vert_set.update(vert_list)
for vert in fix_vert_set:
    sheet.vert_df.loc[vert,'is_active'] = 0

"""Assign dummy edges"""
# cell_20_edges = sheet.edge_df[sheet.edge_df['face'] == 20]
# dummy_index = cell_20_edges.index
# sheet.edge_df.loc[dummy_index,'is_active'] = 0

auto_dummy_edges(sheet)

draw_specs = sheet_spec()
# --- Faces ---
draw_specs['face']['visible'] = True
sheet.face_df['color'] = (sheet.face_df['cell_class'].map({'boundary_fixed': 0.1, 'STB': 0.5}).fillna(0.9))
draw_specs['face']['color'] = sheet.face_df['color']
draw_specs['face']['alpha'] = 0.2
# --- Edges ---
draw_specs['edge']['visible'] = True
sheet.edge_df['width'] = sheet.edge_df['is_active'].eq(0).map({True: 2, False: 0.5})
draw_specs['edge']['width'] = sheet.edge_df['width']

fig, ax = sheet_view(sheet, ['x', 'y'], **draw_specs)
ax.set_title("Steady state before relaxation with dummy")
ax.set_axis_off()
plt.tight_layout()
plt.show()

fig, ax = plot_forces(sheet, geom, model, sheet.coords, scaling=0.5)
ax.set_title("Force plot before relaxation with dummy")
plt.show()

# Use QS solver to start with the steady state of the system.
solver = QSSolver()
res = solver.find_energy_min(sheet, geom, model)
print("Successfull gradient descent? ", res['success'])

draw_specs = sheet_spec()
# --- Faces ---
draw_specs['face']['visible'] = True
sheet.face_df['color'] = (sheet.face_df['cell_class'].map({'boundary_fixed': 0.1, 'STB': 0.5}).fillna(0.9))
draw_specs['face']['color'] = sheet.face_df['color']
draw_specs['face']['alpha'] = 0.2
# --- Edges ---
draw_specs['edge']['visible'] = True
sheet.edge_df['width'] = sheet.edge_df['is_active'].eq(0).map({True: 2, False: 0.5})
draw_specs['edge']['width'] = sheet.edge_df['width']

fig, ax = sheet_view(sheet, ['x', 'y'], **draw_specs)
ax.set_title("Relaxation after dummy assigned")
ax.set_axis_off()
plt.tight_layout()
plt.show()

fig, ax = plot_forces(sheet, geom, model, sheet.coords, scaling=0.5)
ax.set_title("Force plot after relaxation with dummy")
plt.show()



# Generate the cell sheet as three cells.
num_x = 3
num_y = 3
sheet = Sheet.planar_sheet_2d('face', nx = num_x, ny=num_y, distx=0.5, disty=0.5)
geom.update_all(sheet)
# remove non-enclosed faces
sheet.remove(sheet.get_invalid())
drop_face(sheet, 1)
sheet.reset_index(order=True)   #continuous indices in all df, vertices clockwise
fig, ax = sheet_view(sheet)
ax.set_title('Initial setup of the geometry')
plt.show()


# Add dynamics to the model, start with effectors, then change values.
model = model_factory([
    effectors.LineTension,
    effectors.FaceContractility,
    effectors.FaceAreaElasticity
])

specs = {
    'edge': {
        'is_active': 1,
        'line_tension': 10,
        'ux': 0.0,
        'uy': 0.0,
        'uz': 0.0
    },
    'face': {
        'area_elasticity': 110,
        'contractility': 50,
        'is_alive': 1,
        'prefered_area': 2},
    'settings': {
        'grad_norm_factor': 1.0,
        'nrj_norm_factor': 1.0
    },
    'vert': {
        'is_active': 1
    }
}
sheet.vert_df['viscosity'] = 1
# Update the specs (adds / changes the values in the dataframes' columns)
sheet.update_specs(specs, reset=True)
geom.update_all(sheet)

disable_single_vert = 0

if disable_single_vert:
    sheet.vert_df.loc[0, 'is_active'] = 0
    geom.update_all(sheet)

    fig, ax = plot_forces(sheet, geom, model, sheet.coords, scaling=0.1)
    ax.set_title("Force plot of a single cell with one vertex disabled")
    plt.show()

    # Use QS solver to start with the steady state of the system.
    solver = QSSolver()
    res = solver.find_energy_min(sheet, geom, model)
    print("Successfull gradient descent? ", res['success'])

    fig, ax = sheet_view(sheet)
    ax.set_title("Plot of a single cell with disabled vertex after energy minimization")
    plt.show()

else:
    fig, ax = plot_forces(sheet, geom, model, sheet.coords, scaling=0.1)
    ax.set_title("Force plot of a single cell with one vertex disabled")
    plt.show()

    # Use QS solver to start with the steady state of the system.
    solver = QSSolver()
    res = solver.find_energy_min(sheet, geom, model)
    print("Successfull gradient descent? ", res['success'])

    fig, ax = sheet_view(sheet)
    ax.set_title("Plot of a single cell without disabled vertex after energy minimization")
    plt.show()


