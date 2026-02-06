"""
This script contains a vertex model that starts with a bilayer structure.
The system has a multi-class system. Three cellular behaviours are modelled: cell proliferation, fusion and extrusion.
"""

# Load all required modules.
import numpy as np
import re
import matplotlib.pyplot as plt
from tyssue import Sheet
from tyssue.topology.sheet_topology import remove_face
from tyssue.topology.base_topology import close_face, drop_face
from tyssue import PlanarGeometry as geom #for simple 2d geometry
from tyssue.dynamics import effectors, model_factory
from tyssue.dynamics.planar_vertex_model import PlanarModel as smodel
from tyssue.solvers import QSSolver

# 2D plotting
from tyssue.draw import sheet_view
from tyssue.draw.plt_draw import plot_forces
from tyssue.topology.sheet_topology import cell_division
from tyssue.config.draw import sheet_spec

# import my own functions
from my_headers import *
from T3_function import *

import os
from tyssue.io import hdf5 # For saving the datasets
import imageio.v2 as imageio

"""
A cell fusion behaviour function is used when a CT is fusing into the STB layer. The cell class of the selection cell 
should become "STB" at the end of the function.
First, the shared STB edges that share a vertex on the F cells identified.

Secondly, split STB shared vertices that on the outer surface and create separated edges.

Thirdly, split vertex shared by STBs and CT, creating a new CT edge.

Fourthly, F class cell transition to STB class.

Lastly, new dynamic parameters need to be updated to ensure consistent physics rule.
"""
def face_vertices(sheet, face_id):
    """
    Given a face_id, return the list of vertex indices that are part of that face.
    """
    edges = sheet.edge_df[sheet.edge_df['face'] == face_id]
    verts = list(edges['srce']) + list(edges['trgt'])
    return list(set(verts))
def find_local_stb_stb_edge(sheet, F_cell):
    """
    Find the ONE STB–STB mutual edge such that:
    1. Both faces are STB neighbours of F_cell.
    2. At least one endpoint of the edge is a vertex of F_cell.
    Only loops over sheet.sgle_edges.
    Returns a single integer edge index, or None.
    """

    sheet.get_extra_indices()

    # Vertices of the F cell (force into Python ints)
    F_vertices = list(map(int, face_vertices(sheet, F_cell)))

    # STB neighbours of F_cell
    neighbours = sheet.get_neighbors(F_cell)
    stb_neigh = [int(n) for n in neighbours
                 if sheet.face_df.loc[n, 'cell_class'] == 'STB']

    # Loop ONLY over unique edges
    for e in sheet.sgle_edges:

        e = int(e)  # ensure scalar int

        f1 = int(sheet.edge_df.loc[e, 'face'])
        opp = int(sheet.edge_df.loc[e, 'opposite'])

        if opp == -1:
            continue

        f2 = int(sheet.edge_df.loc[opp, 'face'])

        # Condition 1: both faces are STB neighbours of F_cell
        if f1 not in stb_neigh or f2 not in stb_neigh:
            continue

        # Condition 2: edge touches the F cell
        v1 = int(sheet.edge_df.loc[e, 'srce'])
        v2 = int(sheet.edge_df.loc[e, 'trgt'])

        if v1 in F_vertices or v2 in F_vertices:
            return e  # return immediately

    return None


def identify_edge_endpoints(sheet, F_cell, indirect_edge):
    """
    For each edge in local_edges, determine:
    - which endpoint belongs to the F cell
    - which endpoint belongs to the STB neighbour
    Returns a list: [STB_vertex, F_vertex]
    """

    F_vertices = face_vertices(sheet, F_cell)
    v1 = sheet.edge_df.loc[indirect_edge, 'srce']
    v2 = sheet.edge_df.loc[indirect_edge, 'trgt']

    # Determine which vertex belongs to the F cell
    if v1 in F_vertices and v2 not in F_vertices:
        return [v2, v1]

    elif v2 in F_vertices and v1 not in F_vertices:
        return [v1, v2]

    # Return None if neither vertex belongs to the F cell (should not happen if preconditions are met)
    return None

from tyssue.topology.base_topology import split_vert as base_split
from tyssue.topology.sheet_topology import split_vert as sheet_split
from tyssue.topology.sheet_topology import type1_transition

def fuse_single_cell(sheet, F_cell, tau_F_min, tau_F_max):
    """
    Attempt to fuse a CT cell (now in class 'F') into the STB layer.

    Fusion requires a specific geometric configuration:
    - The F cell must touch an STB–STB mutual edge.
    - That edge must share a vertex with the F cell.
    - Only then can the geometric fusion (vertex splitting + T1) proceed.

    If the geometry is NOT ready (e.g., due to T1/T2/T3 transitions or cell division),
    the fusion is postponed by extending the F timer. This prevents:
        - invalid topology operations,
        - isolated STB cells,
        - broken bilayer structure,
        - simulation crashes.

    Parameters
    ----------
    sheet : tyssue.Sheet
        The current tissue sheet.
    F_cell : int
        Index of the cell attempting to fuse.

    Returns
    -------
    new_edge : int or None
        The index of the newly created edge after fusion,
        or None if fusion was postponed.
    """
    if F_cell not in sheet.face_df.index:
        return None
    sse = find_local_stb_stb_edge(sheet, F_cell)
    if sse is None:
        # Geometry not ready for fusion, postpone by extending the timer with a random extra time within F phase.
        extra_time = round(rng.uniform(tau_F_min, tau_F_max), 4)
        sheet.face_df.loc[F_cell, 'timer'] += extra_time
        return None
    # If we reach here, it means the geometry is ready for fusion. Do full geometric operation to fuse the cell.
    stb_face = sheet.edge_df.loc[sse, 'face']
    stbv, fv = identify_edge_endpoints(sheet, F_cell, sse)
    base_split(sheet, stbv, stb_face, sheet.edge_df[sheet.edge_df['face'] == stb_face], epsilon=1, recenter=True)
    new_edge = sheet_split(sheet, fv, F_cell)[0]
    new_edge = type1_transition(sheet, new_edge, do_reindex=True, remove_tri_faces=False, multiplier=5)
    sheet.face_df.loc[F_cell, 'cell_class'] = 'STB'
    geom.update_all(sheet)
    return new_edge

def auto_dummy_edges(sheet):
    sheet.get_extra_indices()
    for i in sheet.edge_df.index:
        opp = sheet.edge_df.loc[i, 'opposite']
        # Boundary edge, always active
        if opp == -1 or opp not in sheet.edge_df.index:
            sheet.edge_df.loc[i, 'is_active'] = 1
            continue

        # Check faces on both sides of the edge
        f1 = sheet.edge_df.loc[i, 'face']
        f2 = sheet.edge_df.loc[opp, 'face']

        # If faces are missing (during topology changes), keep edges active
        if f1 not in sheet.face_df.index or f2 not in sheet.face_df.index:
            sheet.edge_df.loc[i, 'is_active'] = 1
            if opp in sheet.edge_df.index:
                sheet.edge_df.loc[opp, 'is_active'] = 1
            continue

        # Treat E exactly like STB
        c1 = sheet.face_df.loc[f1, 'cell_class']
        c2 = sheet.face_df.loc[f2, 'cell_class']
        is_stb_like_1 = (c1 == 'STB') or (c1 == 'E')
        is_stb_like_2 = (c2 == 'STB') or (c2 == 'E')

        if is_stb_like_1 and is_stb_like_2:
            # Disable dummy edge
            sheet.edge_df.loc[i, 'is_active'] = 0
            sheet.edge_df.loc[opp, 'is_active'] = 0
        else:
            # Enable normal edge
            sheet.edge_df.loc[i, 'is_active'] = 1
            sheet.edge_df.loc[opp, 'is_active'] = 1

    print('Dummy edges updated based on current cell classes.')

def update_draw_specs(sheet, draw_specs):
    """
    Update drawing specifications for faces and edges based on:
    - cell_class (STB vs CT)
    - is_active (dummy edges vs real edges)
    """

    # --- FACE COLORS ---
    # STB = pale yellow (0.7), CT = light purple (0.1)
    sheet.face_df['color'] = sheet.face_df['cell_class'].map(
        lambda c: 0.7 if c == 'STB' else (0.5 if c == 'E' else 0.1)
    )

    draw_specs['face']['color'] = sheet.face_df['color']
    draw_specs['face']['visible'] = True
    draw_specs['face']['alpha'] = 0.2   # transparency

    # --- EDGE WIDTHS ---
    # inactive (dummy) edges = thick, active edges = thin
    sheet.edge_df['width'] = sheet.edge_df['is_active'].map(
        lambda a: 2 if a == 0 else 0.5
    )

    draw_specs['edge']['width'] = sheet.edge_df['width']

    print('Drawing specifications updated based on current cell classes and edge activity.')

def stb_ct_interface_length(sheet):
    length = 0.0
    for e in sheet.edge_df.index:
        f1 = sheet.edge_df.loc[e, 'face']
        opp = sheet.edge_df.loc[e, 'opposite']
        if opp == -1:
            continue
        f2 = sheet.edge_df.loc[opp, 'face']

        if (f1 in sheet.face_df.index and f2 in sheet.face_df.index and
            sheet.face_df.loc[f1, 'cell_class'] != sheet.face_df.loc[f2, 'cell_class']):
            length += sheet.edge_df.loc[e, 'length']
    return length

def stb_detach(sheet, geom, cell_id):
    if cell_id not in sheet.face_df.index:
        return
    sheet.get_extra_indices()
    while True:
        internal_edges = sheet.edge_df[(sheet.edge_df['face'] == cell_id) & (sheet.edge_df['opposite'] != -1)]
        did_t1 = False
        for edge_id in internal_edges.index:
            opposite_edge_id = internal_edges.loc[edge_id, 'opposite']
            opposite_cell = sheet.edge_df.loc[opposite_edge_id, 'face']
            if sheet.face_df.loc[opposite_cell, 'cell_class'] == 'STB' or sheet.face_df.loc[opposite_cell, 'cell_class'] == 'E':
                continue
            else:
                print(f'processing edge {edge_id} for detachment of cell {cell_id}. ')
                collapse_edge(sheet, edge_id, reindex=True)
                geom.update_all(sheet)
                sheet.reset_index(order=False)
                did_t1 = True
                break
        if not did_t1:
            break

def face_boundary_edges(sheet, face_id):
    """Return all boundary edges belonging to a given face."""
    return sheet.edge_df[
        (sheet.edge_df['face'] == face_id) &
        (sheet.edge_df['opposite'] == -1)
    ].index.tolist()


def stb_extrusion(sheet, cell_id):
    if cell_id not in sheet.face_df.index:
        return
    boundary_edges = face_boundary_edges(sheet,cell_id)
    for edge_id in boundary_edges:
        collapse_edge(sheet, edge_id, reindex=False)
    sheet.reset_index(order=False) # Removes redundant indices without reordering.


# Define the directory name
frames_dir = "frames"
# Create directory for frames
if not os.path.exists(frames_dir):
    print(f"Directory '{frames_dir}' does not exist. Creating it.")
    os.makedirs(frames_dir)
else:
    print(f"Directory '{frames_dir}' already exists. Using existing folder.")

# Seed the random number generator
rng = np.random.default_rng(70)


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
plt.show()
ax.set_title("Initial Bilayer Setup")  # Adding title
ax.set_axis_off()
print('Initial geometry plot generated. \n')

# Add a new attribute to the face_df, called "cell class"
sheet.face_df['cell_class'] = 'default'
sheet.face_df['timer'] = np.nan
total_cell_num = len(sheet.face_df)
# Min and Max values for different phase time.
# I am using 1 hour = 0.01 time unit in the simulation, thus 1 full time unit is 100 hours, about 4.17 days.
tau_G1_min = 0.05   # Min G1 phase time is 5 hours
tau_G1_max = 0.11   # Max G1 phase time is 11 hours
tau_S_min = 0.07    # Min S phase time is 7 hours
tau_S_max = 0.08    # Max S phase time is 8 hours
tau_G2_min = 0.03   # Min G2 phase time is 3 hours
tau_G2_max = 0.04   # Max G2 phase time is 4 hours
tau_M_min = 0.005   # Min M phase time is 0.5 hours
tau_M_max = 0.01    # Max M phase time is 1 hour
tau_F_min = 0.24    # Min F phase time is 24 hours
tau_F_max = 0.30     # Max F phase time is 30 hours
stb_age = 0.5

print('New attributes: cell_class; timer created for all cells. \n ')

for i in range(0,num_x-2):  # These are the indices of the bottom layer.
    # All CTs assigned with class ‘G1’, ‘S’, ‘M’, or ‘G2’ based on probabilities that reflect typical times in each stage of the cell cycle
    # Draw a random number between 0 and 1, it's G1 if  < 11/24, S if < 19/24, M if < 20/24, else, G2.
    random_num = rng.random()
    if random_num < 11/24:
        sheet.face_df.loc[i,'cell_class'] = 'G1'
        sheet.face_df.loc[i, 'timer'] = round(rng.uniform(tau_G1_min, tau_G1_max), 4)
    elif 11/24 <= random_num < 19/24:
        sheet.face_df.loc[i,'cell_class'] = 'S'
        sheet.face_df.loc[i, 'timer'] = round(rng.uniform(tau_S_min, tau_S_max), 4)
    elif 19/24 <= random_num < 20/24:
        sheet.face_df.loc[i,'cell_class'] = 'M'
        sheet.face_df.loc[i, 'timer'] = round(rng.uniform(tau_M_min, tau_M_max), 4)
    else:
        sheet.face_df.loc[i,'cell_class'] = 'G2'
        sheet.face_df.loc[i, 'timer'] = round(rng.uniform(tau_G2_min, tau_G2_max), 4)

for i in range(num_x-2,len(sheet.face_df)):     # These are the indices of the top layer.
    sheet.face_df.loc[i,'cell_class'] = 'STB'
    sheet.face_df.loc[i, 'timer'] = round(rng.uniform(0, 0.24), 4)   # Assign a random age to each STB cell, between 0 and the defined maximum age of STB.

print(f'There are {total_cell_num} total cells; equally split into "G1" and "STB" classes. ')

# Add dynamics to the model.
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
        'contractility': 0,
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
sheet.vert_df['viscosity'] = 1.0
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
res = solver.find_energy_min(sheet, geom, smodel)
print("Successfull gradient descent? ", res['success'])

# Deactivate the edges between STB units.
for i in sheet.edge_df.index:
    if sheet.edge_df.loc[i,'opposite'] != -1:
        associated_cell = sheet.edge_df.loc[i,'face']
        opposite_edge = sheet.edge_df.loc[i,'opposite']
        opposite_cell = sheet.edge_df.loc[opposite_edge,'face']
        if sheet.face_df.loc[associated_cell,'cell_class'] == 'STB' and sheet.face_df.loc[opposite_cell,'cell_class'] == 'STB':
            sheet.edge_df.loc[i,'is_active'] = 0
            sheet.edge_df.loc[opposite_edge,'is_active'] = 0

# Create force plot
fig, ax = plot_forces(sheet, geom, model, ['x', 'y'], scaling=0.01)
plt.show()

# Next, I need to colour STB and others differently and bold the dummy edges when plotting.
draw_specs = sheet_spec()
# Enable face visibility.
draw_specs['face']['visible'] = True
for i in sheet.face_df.index:   # Assign face colour based on cell type.
    if sheet.face_df.loc[i,'cell_class'] == 'STB': sheet.face_df.loc[i,'color'] = 0.7
    else: sheet.face_df.loc[i,'color'] = 0.1
draw_specs['face']['color'] = sheet.face_df['color']
draw_specs['face']['alpha'] = 0.2   # Set transparency.

# Enable edge visibility
draw_specs['edge']['visible'] = True
for i in sheet.edge_df.index:
    if sheet.edge_df.loc[i,'is_active'] == 0: sheet.edge_df.loc[i,'width'] = 2
    else: sheet.edge_df.loc[i,'width'] = 0.5
draw_specs['edge']['width'] = sheet.edge_df['width']

fig, ax = sheet_view(sheet, ['x', 'y'], **draw_specs)
ax.set_axis_off()
plt.show()


# Set the threshold values for mesh restructure.
t1_threshold = sheet.edge_df['length'].mean()/10
t2_threshold = sheet.face_df['area'].mean()/10
d_min = t1_threshold
d_sep = d_min *1.5
max_movement = t1_threshold / 2
# Before the simulation loop:
fusion_events = []      # number of fusion events at each time step
time_list = []
STB_area = []
# Also keep the record for initial stb area, stb-ct interface length and stb mean thickness.
initial_stb_area = sheet.face_df.loc[sheet.face_df['cell_class'] == 'STB', 'area'].sum()
initial_stb_ct_interface_length = stb_ct_interface_length(sheet)
initial_stb_thickness = initial_stb_area/initial_stb_ct_interface_length

# Start simulating.
t = 0
t_end = 0.5

while t <= t_end:
    dt = 0.001  # initial time step, will be updated dynamically later.

    # Mesh restructure check
    # T1 transition, edge rearrangment check
    while True:
        # Check for any edge below the threshold, starting from index 0 upwards
        edge_to_process = None
        # Clean up the vertex mesh to make sure all polygons are valid.
        invalid_edges = sheet.get_invalid()
        unclosed_faces = list(set(sheet.edge_df.loc[invalid_edges, 'face']))
        for face in unclosed_faces:
            try:
                close_face(sheet, face)
            except ValueError:
                pass
        geom.update_all(sheet)
        for index in sheet.edge_df.index:
            if sheet.edge_df.loc[index, 'length'] < t1_threshold:
                # Adding safeguard to skip malformed transitions
                srce = sheet.edge_df.loc[index, 'srce']
                trgt = sheet.edge_df.loc[index, 'trgt']
                # Check for duplicate edges that would cause T1 to break topology
                edge_face = sheet.edge_df.loc[index, 'face']
                is_duplicate = (
                                       (sheet.edge_df['face'] == edge_face) &
                                       (sheet.edge_df['srce'] == srce) &
                                       (sheet.edge_df['trgt'] == trgt)
                               ).sum() > 1

                if is_duplicate:
                    print(f"Skipping edge {index} due to duplicate srce-trgt-face entry.")
                    continue
                edge_to_process = index
                edge_length = sheet.edge_df.loc[edge_to_process, 'length']
                # print(f'Edge {edge_to_process} is too short: {edge_length}')
                # Process the identified edge with T1 transition
                type1_transition(sheet, edge_to_process, remove_tri_faces=False, multiplier=2)
                # Post-processing the mesh after a T1 transition
                sheet.reset_index(order=True)
                geom.update_all(sheet)
                sheet.remove(sheet.get_invalid()) # clean up bad faces/edges
                sheet.get_extra_indices()
                break
                # Exit the loop if no edges are below the threshold
        if edge_to_process is None:
            break
    geom.update_all(sheet)

    # T2 transition check.
    tri_faces = sheet.face_df[(sheet.face_df["num_sides"] < 4) &
                              (sheet.face_df["area"] < t2_threshold)].index
    while len(tri_faces):
        remove_face(sheet, tri_faces[0])
        # Recompute the list of triangular faces below the area threshold after each removal
        tri_faces = sheet.face_df[(sheet.face_df["num_sides"] < 4) &
                                  (sheet.face_df["area"] < t2_threshold)].index
    sheet.reset_index(order=True)
    geom.update_all(sheet)

    # T3 transition.
    while True:
        T3_todo = None
        # print('computing boundary indices.')
        boundary_vert, boundary_edge = find_boundary(sheet)

        for edge_e in boundary_edge:
            # Extract source and target vertex IDs
            srce_id, trgt_id = sheet.edge_df.loc[edge_e, ['srce', 'trgt']]
            for vertex_v in boundary_vert:
                if vertex_v == srce_id or vertex_v == trgt_id:
                    continue

                distance, nearest = dist_computer(sheet, edge_e, vertex_v, d_sep)
                if distance < d_min:
                    T3_todo = vertex_v
                    # print(f'Found incoming vertex: {vertex_v} and colliding edge: {edge_e}')
                    T3_swap(sheet, edge_e, vertex_v, nearest, d_sep)
                    sheet.reset_index(order=False)
                    geom.update_all(sheet)
                    sheet.remove(sheet.get_invalid())
                    sheet.get_extra_indices()
                    # fig, ax = sheet_view(sheet, edge={'head_width': 0.1})
                    # for face, data in sheet.vert_df.iterrows():
                    #     ax.text(data.x, data.y, face)
                    break
            if T3_todo is not None:
                break  # Exit outer loop to restart with updated boundary
        if T3_todo is None:
            break
        sheet.reset_index(order=True)
        geom.update_all(sheet)
        sheet.remove(sheet.get_invalid())
        sheet.get_extra_indices()

    # For all mature "S" cells, it is possible for them to proliferate; fuse or quiescent.
    S_cells = sheet.face_df.index[sheet.face_df['cell_class'] == 'S'].tolist()
    fusion_count = 0
    for cell in S_cells:
        can_fuse = 0
        neighbours = sheet.get_neighbors(cell)
        for i in neighbours:
            if sheet.face_df.loc[i, 'cell_class'] == 'STB':
                can_fuse = 1
            else:
                continue
        # Use rng to randomly generate a number between 1 and 10, this will determine the fate of the mature CT.
        cell_fate_roulette = rng.random()
        if can_fuse == 1 and cell_fate_roulette < 0.2:  # If CT is adjacent to STB, then it has 20% probability to fuse.
            sheet.face_df.loc[cell, 'cell_class'] = 'F'
            # Add a timer for each cell enters 'F'.
            sheet.face_df.loc[cell, 'timer'] = round(rng.uniform(tau_F_min, tau_F_max), 4)
            fusion_count += 1
        elif can_fuse == 1 and 0.2< cell_fate_roulette <0.3: # If CT is adjacent to STB, it has 10% probability to divide.
            sheet.face_df.loc[cell, 'cell_class'] = 'G2'
            sheet.face_df.loc[cell, 'timer'] = round(rng.uniform(tau_G2_min, tau_G2_max), 4)

        elif cell_fate_roulette <= 0.3:  # If CT is not adjacent to STB, then divide with probability 30%.
            sheet.face_df.loc[cell, 'cell_class'] = 'G2'
            # Add a timer for each cell enters "G2".
            sheet.face_df.loc[cell, 'timer'] = round(rng.uniform(tau_G2_min, tau_G2_max), 4)
        else:
            continue
    geom.update_all(sheet)

    # At the end of the timer, "G2" becomes "M".
    G2_cells = sheet.face_df.index[sheet.face_df['cell_class'] == 'G2'].tolist()
    for cell in G2_cells:
        if sheet.face_df.loc[cell, 'timer'] < 0:
            sheet.face_df.loc[cell, 'cell_class'] = 'M'
            sheet.face_df.loc[cell, 'timer'] = round(rng.uniform(tau_M_min, tau_M_max), 4)
        else:
            sheet.face_df.loc[cell, 'timer'] -= dt

    geom.update_all(sheet)

    # Cell division.
    # For all cells in "M", divide the cell. Then cells become "G1".
    # Store the centroid before iteration of cells.
    unique_edges_df = sheet.edge_df.drop_duplicates(subset='face')
    centre_data = unique_edges_df.loc[:, ['face', 'fx', 'fy']]
    # Cells in "M" class can be divided.
    cells_can_divide = sheet.face_df.index[sheet.face_df['cell_class'] == 'M'].tolist()
    for index in cells_can_divide:
        daughter_index = division_mt(sheet, rng=rng, cent_data=centre_data, cell_id=index)
        sheet.face_df.loc[index, 'cell_class'] = 'G1'
        sheet.face_df.loc[daughter_index, 'cell_class'] = 'G1'
        # Add a timer for each cell enters "G1".
        sheet.face_df.loc[index, 'timer'] = round(rng.uniform(tau_G1_min, tau_G1_max), 4)
        sheet.face_df.loc[daughter_index, 'timer'] = round(rng.uniform(tau_G1_min, tau_G1_max), 4)

    geom.update_all(sheet)

    # At the end of the timer, "G1" class becomes "S".
    G1_cells = sheet.face_df.index[sheet.face_df['cell_class'] == 'G1'].tolist()
    for cell in G1_cells:
        if sheet.face_df.loc[cell, 'timer'] < 0:
            sheet.face_df.loc[cell, 'cell_class'] = 'S'
            sheet.face_df.loc[cell, 'timer'] = round(rng.uniform(tau_S_min, tau_S_max), 4)
        else:
            sheet.face_df.loc[cell, 'timer'] -= dt

    sheet.reset_index(order=True)
    geom.update_all(sheet)

    # At the end of a timer, "F" class becomes "STB" and dummy edge is generated.
    F_cells = sheet.face_df.index[sheet.face_df['cell_class'] == 'F'].tolist()
    for cell in F_cells:
        if sheet.face_df.loc[cell, 'timer'] < 0:
            fuse_single_cell(sheet, cell, tau_F_min, tau_F_max)
        else:
            sheet.face_df.loc[cell, 'timer'] -= dt
    geom.update_all(sheet)

    # Extrude the 'E' units before assigning new 'E' units.
    E_units = sheet.face_df.index[sheet.face_df['cell_class'] == 'E'].tolist()
    for unit in E_units:
        stb_extrusion(sheet, unit)
    geom.update_all(sheet)

    # Work on STB units
    STB_units = sheet.face_df.index[sheet.face_df['cell_class'] == 'STB'].tolist()
    for unit in STB_units:
        if sheet.face_df.loc[unit, 'timer'] > stb_age:
            if rng.random() < 0.1:  # 10% probability to extrude.
                sheet.face_df.loc[unit, 'cell_class'] = 'E'  # Mark the cell as extruding.
                stb_detach(sheet, geom, unit)
        else:
            sheet.face_df.loc[unit, 'timer'] += dt
    geom.update_all(sheet)

    # Update dummy edges after all cell class changes.
    auto_dummy_edges(sheet)

    # Force computing and updating positions.
    valid_active_verts = sheet.active_verts[sheet.active_verts.isin(sheet.vert_df.index)]
    pos = sheet.vert_df.loc[valid_active_verts, sheet.coords].values
    # get the movement of position based on dynamical dt.
    dt, movement = time_step_bot(sheet, dt, max_dist_allowed=max_movement)
    new_pos = pos + movement
    dt = Decimal(dt)
    # Save the new positions back to `vert_df`
    sheet.vert_df.loc[valid_active_verts, sheet.coords] = new_pos
    geom.update_all(sheet)

    # Tracking STB Area.
    real_time_hours = t * 100
    total_STB = sheet.face_df.loc[sheet.face_df['cell_class'] == 'STB', 'area'].sum()
    STB_area.append(total_STB)
    time_list.append(real_time_hours)
    # Record fusion events and time.
    fusion_events.append(fusion_count)
    # Print time in console.
    print(f'At time {real_time_hours:.1f} hours\n')

    # Generate the plot at this time step.
    update_draw_specs(sheet, draw_specs)  # Update drawing specifications based on current sheet state
    fig, ax = sheet_view(sheet, ['x', 'y'], **draw_specs)
    ax.title.set_text(f'time = {real_time_hours:.1f}')
    ax.set_axis_off()
    # Save to file instead of showing.
    frame_path = f"frames/frame_{real_time_hours:.1f}.png"
    plt.savefig(frame_path)
    plt.close(fig)  # Close figure to prevent memory leaks

    # Update time_point
    t += dt

final_stb_area = sheet.face_df.loc[sheet.face_df['cell_class'] == 'STB', 'area'].sum()
final_stb_ct_interface_length = stb_ct_interface_length(sheet)
final_stb_thickness = final_stb_area/final_stb_ct_interface_length

# Write the final sheet to a hdf5 file.
hdf5.save_datasets('PFE_50h.hdf5', sheet)

""" Generate the video based on the frames saved. """
# Path to folder containing the frame images
frame_folder = "frames"

# Helper function to extract the numeric part from a filename
# For example, from "frame_12.png", it extracts 12
def extract_number(fname):
    match = re.search(r'\d+', fname)
    return int(match.group()) if match else -1  # If no number found, use -1

# List and numerically sort all .png files in the frame folder
frame_files = sorted([
    os.path.join(frame_folder, fname)
    for fname in os.listdir(frame_folder)
    if fname.endswith('.png')  # Only include PNG files
], key=lambda x: extract_number(os.path.basename(x)))  # Sort by extracted number

# Create a video with 15 frames per second, change the name to whatever you want the name of mp4 to be.
with imageio.get_writer('PFE_50h.mp4', fps=15, format='ffmpeg') as writer:
    # Read and append each frame in sorted order
    for filename in frame_files:
        image = imageio.imread(filename)  # Load image from the folder
        writer.append_data(image)        # Write image to video



plt.figure(figsize=(8, 5))
plt.plot(time_list, STB_area, label='Total STB Area', color='purple')
plt.xlabel('Time')
plt.ylabel('STB Area')
plt.title('STB Area Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(time_list, fusion_events, label='Fusion events per time step', color='red')
plt.xlabel('Time')
plt.ylabel('Number of fusion events')
plt.title('Fusion Events Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

hours_per_step = dt * 100
fusion_rate = [count / hours_per_step for count in fusion_events]

plt.figure(figsize=(8, 5))
plt.plot(time_list, fusion_rate, color='darkred')
plt.xlabel('Time')
plt.ylabel('Fusion events per hour')
plt.title('Fusion Rate Over Time')
plt.grid(True)
plt.tight_layout()
plt.show()

# Create a DataFrame with all tracked quantities
import pandas as pd

df = pd.DataFrame({
    "time": time_list,
    "fusion_events": fusion_events,
    "STB_area": STB_area
})
# Save to CSV
df.to_csv("simulation_output_50h_pf.csv", index=False)
print("Saved simulation_output.csv")

print(f'The initial STB area is {initial_stb_area:.2f},\n the initial STB-CT interface length is {initial_stb_ct_interface_length:.2f},\n and the initial mean thickness is {initial_stb_thickness:.2f}.\n')


print('\n This is the end of this script. (＾• ω •＾) ')
