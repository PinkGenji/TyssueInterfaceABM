"""
This script works as a lab bench for
"""

# Load all required modules.
import random
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import correspond
from tyssue import Sheet, History, PlanarGeometry
from tyssue.topology.sheet_topology import remove_face
from tyssue.dynamics import effectors, model_factory
from tyssue.behaviors import EventManager
from tyssue.solvers.viscous import EulerSolver



# 2D plotting
from tyssue.draw import sheet_view
from tyssue.config.draw import sheet_spec

# import my own functions
from my_headers import *
from T3_function import *



# Moving function definitions to the top of the file
def drop_face(sheet, face, **kwargs):
    """
    Removes the face indexed by "face" and all associated edges
    """
    edge = sheet.edge_df.loc[(sheet.edge_df['face'] == face)].index
    print(f"Dropping face '{face}'")
    sheet.remove(edge, **kwargs)


def cell_cycle_transition(sheet, manager, dt, cell_id=0, p_recruit=0.1, G2_duration=0.4, G1_duration=0.11):
    """
    Controls cell class state transitions for cell cycle based on timers and probabilities.

    Parameters
    ----------
    sheet: tyssue.Sheet
        The tissue sheet.
    manager: EventManager
        The event manager scheduling the behaviour.
    cell_id: Integer
        ID of the cell being controlled.
    p_recruit: float
        Probability for an 'S' cell to be recruited to 'G2'.
    dt: float
        Time step increment.
    G2_duration: float
        Fixed duration cells stay in G2 phase.
    G1_duration: float
        Fixed duration cells stay in G1 phase.
    """

    # Record the current cell class
    current_class = sheet.face_df.loc[cell_id, 'cell_class']
    # (1) Recruit mature 'S' cells into G2 with probability p_recruit
    if current_class == 'S':
        if np.random.rand() < p_recruit:
            sheet.face_df.loc[cell_id, 'cell_class'] = 'G2'
            sheet.face_df.loc[cell_id, 'timer'] = G2_duration
        # append to next deque
        manager.append(cell_cycle_transition, dt=dt, cell_id=cell_id)

    # (2) Decrement timers for cells in G2; when timer ends, move to M
    elif current_class == 'G2':
        sheet.face_df.loc[cell_id, 'timer'] -= dt
        if sheet.face_df.loc[cell_id, 'timer'] <= 0:
            sheet.face_df.loc[cell_id, 'cell_class'] = 'M'
        # append to next deque
        manager.append(cell_cycle_transition, dt=dt, cell_id=cell_id)

    # (3) For cells in M, perform division and set daughters to G1 with timer
    elif current_class == 'M':
        centre_data = sheet.edge_df.drop_duplicates(subset='face')[['face', 'fx', 'fy']]
        daughter = division_mt(sheet, rng, centre_data, cell_id)
        # Set parent and daughter to G1 with G1 timer
        sheet.face_df.loc[cell_id, 'cell_class'] = 'G1'
        sheet.face_df.loc[daughter, 'cell_class'] = 'G1'
        sheet.face_df.loc[cell_id, 'timer'] = G1_duration
        sheet.face_df.loc[daughter, 'timer'] = G1_duration
        # append to next deque
        manager.append(cell_cycle_transition, dt=dt, cell_id=cell_id)
        manager.append(cell_cycle_transition, dt=dt, cell_id=daughter)

    # (4) Decrement timers for G1 cells; when timer ends, move to S
    elif current_class == 'G1':
        sheet.face_df.loc[cell_id, 'timer'] -= dt
        if sheet.face_df.loc[cell_id, 'timer'] <= 0:
            sheet.face_df.loc[cell_id, 'cell_class'] = 'S'
        # append to next deque
        manager.append(cell_cycle_transition, dt=dt, cell_id=cell_id)


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

# Add a new attribute to the face_df, called "cell class"
sheet.face_df['cell_class'] = 'default'
sheet.face_df['timer'] = 'NA'
total_cell_num = len(sheet.face_df)
print('New attributes: cell_class; timer created for all cells. \n ')

for i in range(0, num_x - 2):  # These are the indices of the bottom layer.
    sheet.face_df.loc[i, 'cell_class'] = 'G1'
    # Add a timer for each cell enters "G1".
    sheet.face_df.loc[i, 'timer'] = 0.11

for i in range(num_x - 2, len(sheet.face_df)):  # These are the indices of the top layer.
    sheet.face_df.loc[i, 'cell_class'] = 'STB'

# Apply drawing specs, so STB and CT have different colours.
draw_specs = sheet_spec()
# Enable face visibility.
draw_specs['face']['visible'] = True
for i in sheet.face_df.index:  # Assign face colour based on the cell class.
    if sheet.face_df.loc[i, 'cell_class'] == 'STB':
        sheet.face_df.loc[i, 'color'] = 0.7
    else:
        sheet.face_df.loc[i, 'color'] = 0.1
draw_specs['face']['color'] = sheet.face_df['color']
draw_specs['face']['alpha'] = 0.2  # Set transparency.

# Plot the figure to see the initial setup is what we want.
fig, ax = sheet_view(sheet, **draw_specs)
plt.show()
ax.set_title("Initial Bilayer Setup")  # Adding title
print('Initial geometry plot generated. \n')
print(f'There are {total_cell_num} total cells; equally split into "G1" and "STB" classes. \n')

# Setup what effectors we want for the model
model = model_factory([effectors.LineTension, effectors.FaceContractility, effectors.FaceAreaElasticity])

# specify the values of parameters to the model.
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
# Update the specs to the dataframes.
sheet.update_specs(specs, reset=True)
sheet.get_opposite()
geom.update_all(sheet)

# Adjust for cell-boundary adhesion force.
for i in sheet.edge_df.index:
    if sheet.edge_df.loc[i, 'opposite'] == -1:
        sheet.edge_df.loc[i, 'line_tension'] *= 2
    else:
        continue
geom.update_all(sheet)

# Set the value of constants for mesh restructure, which are parts of my own solver in loop.
t1_threshold = sheet.edge_df.length.mean() / 10
t2_threshold = sheet.face_df['area'].mean() / 10
d_min = t1_threshold
d_sep = d_min * 1.5

# Start simulating.
# Initialise the Event Manager for Tyssue's Built-in Euler Solver.
manager = EventManager('face')
history = History(sheet)  # Record the initial configuration of the sheet.

i = 0
# Only one iteration
while i < 1:
    dt = 0.0005
    # Mesh restructure check
    # T1 transition, edge rearrangment check
    while True:
        # Check for any edge below the threshold, starting from index 0 upwards
        edge_to_process = None
        for index in sheet.edge_df.index:
            if sheet.edge_df.loc[index, 'length'] < t1_threshold:
                # Adding safeguard to skip malformed transitions
                srce = sheet.edge_df.loc[index, 'srce']
                trgt = sheet.edge_df.loc[index, 'trgt']
                # Ensure both vertices are part of exactly 2 faces (simple topology)
                if (
                        (sheet.edge_df['srce'] == srce).sum() > 5 or
                        (sheet.edge_df['trgt'] == trgt).sum() > 5
                ):
                    print(f"Skipping edge {index} due to weird topology.")
                    continue
                edge_to_process = index
                edge_length = sheet.edge_df.loc[edge_to_process, 'length']
                # Process the identified edge with T1 transition
                type1_transition(sheet, edge_to_process, remove_tri_faces=False, multiplier=1.5)
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
                    T3_swap(sheet, edge_e, vertex_v, nearest, d_sep)
                    sheet.reset_index(order=False)
                    geom.update_all(sheet)
                    sheet.get_extra_indices()
                    break

            if T3_todo is not None:
                break  # Exit outer loop to restart with updated boundary

        if T3_todo is None:
            break

    # Select all mature "S" cells.
    S_cells = sheet.face_df.index[sheet.face_df['cell_class'] == 'S'].tolist()

    for cell in S_cells:
        # Use rng to randomly generate a number between 1 and 10, this will determine the fate of the mature CT.
        cell_fate_roulette = rng.random()
        if cell_fate_roulette <= 0.3:  # Use probability of 0.5 for division.
            sheet.face_df.loc[cell, 'cell_class'] = 'G2'
            # Add a timer for each cell enters "G2".
            sheet.face_df.loc[cell, 'timer'] = 0.4
        else:
            continue
    geom.update_all(sheet)

    # At the end of the timer, "G2" becomes "M".
    G2_cells = sheet.face_df.index[sheet.face_df['cell_class'] == 'G2'].tolist()
    for cell in G2_cells:
        if sheet.face_df.loc[cell, 'timer'] < 0:
            sheet.face_df.loc[cell, 'cell_class'] = 'M'
        else:
            sheet.face_df.loc[cell, 'timer'] -= dt
    geom.update_all(sheet)

    # Cell division.
    # For all cells in "M", divide the cell. Then cells becomes "G1".
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
        sheet.face_df.loc[index, 'timer'] = 0.11
        sheet.face_df.loc[daughter_index, 'timer'] = 0.11
    geom.update_all(sheet)

    # At the end of the timer, "G1" class becomes "S".
    G1_cells = sheet.face_df.index[sheet.face_df['cell_class'] == 'G1'].tolist()
    for cell in G1_cells:
        if sheet.face_df.loc[cell, 'timer'] < 0:
            sheet.face_df.loc[cell, 'cell_class'] = 'S'
        else:
            sheet.face_df.loc[cell, 'timer'] -= dt

    sheet.reset_index(order=True)
    geom.update_all(sheet)

    # Force computing and updating positions using my own solver.
    valid_active_verts = sheet.active_verts[sheet.active_verts.isin(sheet.vert_df.index)]
    pos = sheet.vert_df.loc[valid_active_verts, sheet.coords].values.ravel()
    dt, movement = time_step_bot(sheet, dt, max_dist_allowed= t1_threshold /2 )

    # Now, keep shrinking dt in Tyssue's Euler solver and compare the movement difference
    dt_range = np.linspace(dt, dt/100, 100)
    gaps = []
    for dt in dt_range:
        # Revert the geometry to the initial state, then use Tyssue's built-in Euler Solver.
        sheet = history.retrieve(0)
        # Initialise the Tyssue's Euler solver.
        solver = EulerSolver(sheet, geom, model, manager=manager)
        # Grab the movement and dot_r computed from Tyssue's Euler solver.
        movement2, dot_r2 = solver.single_step_movement(dt=dt)
        # Now, compare movement and movement2
        # max_diff = np.max(np.abs(movement - movement2))
        diff = np.mean(movement-movement2)
        gaps.append(diff)

    # create a fig with one subplot, which is a gaps vs t_range plot.
    fig,ax = plt.subplots(1)
    ax.plot(dt_range, gaps)
    ax.set_title("Plot of movement difference against dt values")
    ax.set_xlabel("Time step (dt)")
    ax.set_ylabel("Movement difference")
    plt.show()
    smallest_gap = np.min(gaps)
    corresponding_dt = dt_range[gaps.index(smallest_gap)]
    print(f'The smallest gap is: {smallest_gap}, when dt is: {corresponding_dt}')

    i += 1  # Finish the iteration

print('\n This is the end of this script. (＾• ω •＾) ')
