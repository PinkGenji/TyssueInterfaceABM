"""
This script is improved from the contact inhibition model with single cell class by implement multi cell classes and
the transitions between different classes.
"""

import matplotlib.pyplot as plt

from tyssue import Sheet, config  # import core object
from tyssue import PlanarGeometry as geom  # for simple 2d geometry

# For cell topology/configuration
from tyssue.topology.sheet_topology import type1_transition
from tyssue.topology.base_topology import collapse_edge, add_vert
from tyssue.topology.sheet_topology import split_vert as sheet_split
from tyssue.topology.bulk_topology import split_vert as bulk_split
from tyssue.topology import condition_4i, condition_4ii

## model and solver
from tyssue.topology.sheet_topology import remove_face

# 2D plotting
from tyssue.draw import sheet_view
from tyssue.draw.plt_draw import plot_forces
# import my own functions
from my_headers import *
from T3_function import *

# Set up the random number generator (RNG)
rng = np.random.default_rng(70)

# Generate a single cell by generate
num_x = 3
num_y = 3
sheet = Sheet.planar_sheet_2d('face', nx=num_x, ny=num_y, distx=1, disty=1)
sheet.remove(sheet.get_invalid())
geom.update_all(sheet)
# Visualize the sheet.
fig, ax = sheet_view(sheet, edge={'head_width': 0.1})
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)
plt.show()
# remove non-enclosed faces
sheet.remove(sheet.get_invalid())
delete_face(sheet, 1)
geom.update_all(sheet)
sheet.reset_index(order=True)  # continuous indices in all df, vertices clockwise
sheet.get_extra_indices()
# # We need to creata a new colum to store the cell cycle time, default a 0, then minus.
# sheet.face_df['T_cycle'] = 0
# Visualize the sheet.
fig, ax = sheet_view(sheet, edge={'head_width': 0.1})
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)
plt.show()

# First, we need a way to compute the energy, then use gradient descent.
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

# Assign the cell to be in class "S".
sheet.face_df['cell_class'] = 'default'
sheet.face_df.loc[0,'cell_class'] = 'S'
print('The only cell is assigned to the "S" class.')

# We need set the all the threshold value first.
t1_threshold = 0.01
t2_threshold = 0.1
d_min = 0.0008
d_sep = 0.011
division_threshold = 1
inhibition_threshold = 0.8
max_movement = t1_threshold / 2
time_stamp = []
cell_counter = []
area_intotal = []
cell_ave_intime = []

# Now assume we want to go from t = 0 to t= 0.2, dt = 0.1
t = Decimal("0")
t_end = Decimal("1")

while t <= t_end:
    dt = 0.001

    # Mesh restructure check
    # T1 transition, edge rearrangment check
    while True:
        # Check for any edge below the threshold, starting from index 0 upwards
        edge_to_process = None
        for index in sheet.edge_df.index:
            if sheet.edge_df.loc[index, 'length'] < t1_threshold:
                edge_to_process = index
                edge_length = sheet.edge_df.loc[edge_to_process, 'length']
                # print(f'Edge {edge_to_process} is too short: {edge_length}')
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
                    sheet.get_extra_indices()
                    fig, ax = sheet_view(sheet, edge={'head_width': 0.1})
                    for face, data in sheet.vert_df.iterrows():
                        ax.text(data.x, data.y, face)
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
        if cell_fate_roulette <= 0.5: # Use probability of 0.5 for division.
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
            if cell == 1:
                print(f'Cell 1 enter "S" at time {t}. ')
        else:
            sheet.face_df.loc[cell, 'timer'] -= dt

    sheet.reset_index(order=True)
    geom.update_all(sheet)

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

    # Add trackers for quantify.
    cell_num_count = len(sheet.face_df)
    S_count = len(sheet.face_df.index[sheet.face_df['cell_class'] == 'S'].tolist())
    M_count = len(sheet.face_df.index[sheet.face_df['cell_class'] == 'M'].tolist())
    G1_count = len(sheet.face_df.index[sheet.face_df['cell_class'] == 'G1'].tolist())
    G2_count = len(sheet.face_df.index[sheet.face_df['cell_class'] == 'G2'].tolist())


    print(f'At time {t}, there are {S_count} S cells; {G2_count} G2 cells; {M_count} M cells; {G1_count} G1 cells. \n')

    # Update time_point
    t += dt
    t = t.quantize(Decimal("0.0001"))  # Keeps t rounded to 5 decimal places

fig, ax = sheet_view(sheet)
ax.title.set_text(f'time = {round(t, 5)}')
plt.show()

print('\n This is the end of this script. (＾• ω •＾) ')
