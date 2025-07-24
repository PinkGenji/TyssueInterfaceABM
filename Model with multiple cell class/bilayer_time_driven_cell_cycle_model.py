"""
This script models a time driven (and only time) cell cycle started with a bilayer geometry.
Following are the aims:
    (1) Mature CTs Cell gets recruited from mature into G2 with simple probability p.
    (2) Cell does not grow but is programmed to stay in G2 for a fixed period of time.
    (3) Cell moves to M phase and divides.
    (4) Two daughter cells now in G1 don’t actively grow, but instead are programmed to stay in G1 for fixed amount of time.
    (5) Cells move from G1 to S.
    Note: the only 'cell growth' factor is the target area term in the energy function.
"""


# Load all required modules.
import random
import numpy as np
import re
import matplotlib.pyplot as plt
from tyssue import Sheet, History, PlanarGeometry
from tyssue.topology.sheet_topology import remove_face
from tyssue.dynamics import effectors, model_factory
from tyssue.behaviors import EventManager
from tyssue.behaviors.sheet import basic_events
from tyssue.solvers.viscous import EulerSolver

# 2D plotting
from tyssue.draw import sheet_view
from tyssue.topology.sheet_topology import cell_division

# import my own functions
from my_headers import *
from T3_function import *

import os
import imageio.v2 as imageio

# Define the directory name
frames_dir = "frames"
# Create directory for frames
if not os.path.exists(frames_dir):
    print(f"Directory '{frames_dir}' does not exist. Creating it.")
    os.makedirs(frames_dir)
else:
    print(f"Directory '{frames_dir}' already exists. Using existing folder.")

random.seed(42)  # Controls Python's random module (e.g. event shuffling)
np.random.seed(42) # Controls NumPy's RNG (e.g. vertex positions, topology)
rng = np.random.default_rng(70)    # Seed the random number generator for my own division function.

Tyssue_Euler_solver = False # control which solver to use.

# Generate the initial cell sheet for bilayer.
geom = PlanarGeometry
print('\n Now we change the initial geometry to bilayer.')
num_x = 16
num_y = 4

sheet =Sheet.planar_sheet_2d(identifier='bilayer', nx = num_x, ny = num_y, distx = 1, disty = 1)
geom.update_all(sheet)
#Updates the sheet geometry by updating: * the edge vector coordinates * the edge lengths * the face centroids
# * the normals to each edge associated face * the face areas.

# remove non-enclosed faces
sheet.remove(sheet.get_invalid())

def drop_face(sheet, face, **kwargs):
    """
    Removes the face indexed by "face" and all associated edges
    """
    edge = sheet.edge_df.loc[(sheet.edge_df['face'] == face)].index
    print(f"Removing face '{face}'")
    sheet.remove(edge, **kwargs)

# Repeatedly remove all non-hexagonal faces until none remain
while np.any(sheet.face_df['num_sides'].values != 6):
    bad_face = sheet.face_df[sheet.face_df['num_sides'] != 6].index[0]
    drop_face(sheet, bad_face)

geom.update_all(sheet)
sheet.get_opposite()

# Plot the figure to see the initial setup is what we want.
fig, ax = sheet_view(sheet)
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)
plt.show()
ax.set_title("Initial Bilayer Setup")  # Adding title

print('Initial geometry plot generated. \n')

model = model_factory([
    effectors.LineTension,
    effectors.FaceContractility,
    effectors.FaceAreaElasticity
])

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
sheet.get_opposite()
geom.update_all(sheet)

# Adjust for cell-boundary adhesion force.
for i in sheet.edge_df.index:
    if sheet.edge_df.loc[i, 'opposite'] == -1:
        sheet.edge_df.loc[i, 'line_tension'] *= 2
    else:
        continue
geom.update_all(sheet)


# Add a new attribute to the face_df, called "cell class"
sheet.face_df['cell_class'] = 'default'
sheet.face_df['timer'] = 'NA'
total_cell_num = len(sheet.face_df)
print('New attributes: cell_class; timer created for all cells. \n ')

for i in range(0,num_x-2):  # These are the indices of the bottom layer.
    sheet.face_df.loc[i,'cell_class'] = 'G1'
    # Add a timer for each cell enters "G1".
    sheet.face_df.loc[i, 'timer'] = 0.11

for i in range(num_x-2,len(sheet.face_df)):     # These are the indices of the top layer.
    sheet.face_df.loc[i,'cell_class'] = 'STB'

print(f'There are {total_cell_num} total cells; equally split into "G1" and "STB" classes. ')

def cell_cycle_transition(sheet, manager, cell_id=0, p_recruit=0.1, dt=0.1, G2_duration=0.4, G1_duration=0.11):
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
    print(f'Cell {cell_id} is being checked by cell cycle ')
    current_class = sheet.face_df.loc[cell_id,'cell_class']
    # (1) Recruit mature 'S' cells into G2 with probability p_recruit
    if current_class == 'S':
        if np.random.rand() < p_recruit:
            sheet.face_df.loc[cell_id, 'cell_class'] = 'G2'
            sheet.face_df.loc[cell_id, 'timer'] = G2_duration
        # append to next deque
        manager.append(cell_cycle_transition, cell_id=cell_id)

    # (2) Decrement timers for cells in G2; when timer ends, move to M
    elif current_class == 'G2':
        sheet.face_df.loc[cell_id, 'timer'] -= dt
        if sheet.face_df.loc[cell_id, 'timer'] <= 0:
            sheet.face_df.loc[cell_id, 'cell_class'] = 'M'
        # append to next deque
        manager.append(cell_cycle_transition, cell_id=cell_id)

    # (3) For cells in M, perform division and set daughters to G1 with timer
    elif current_class == 'M':
        centre_data = sheet.edge_df.drop_duplicates(subset='face')[['face', 'fx', 'fy']]
        daughter = division_mt(sheet,rng, centre_data, cell_id)
        # Set parent and daughter to G1 with G1 timer
        sheet.face_df.loc[cell_id, 'cell_class'] = 'G1'
        sheet.face_df.loc[daughter, 'cell_class'] = 'G1'
        sheet.face_df.loc[cell_id, 'timer'] = G1_duration
        sheet.face_df.loc[daughter, 'timer'] = G1_duration
        # append to next deque
        manager.append(cell_cycle_transition, cell_id=cell_id)
        manager.append(cell_cycle_transition, cell_id = daughter)

    # (4) Decrement timers for G1 cells; when timer ends, move to S
    elif current_class == 'G1':
        sheet.face_df.loc[cell_id, 'timer'] -= dt
        if sheet.face_df.loc[cell_id, 'timer'] <= 0:
            sheet.face_df.loc[cell_id, 'cell_class'] = 'S'
        # append to next deque
        manager.append(cell_cycle_transition, cell_id=cell_id)


# Initialise the Event Manager
manager = EventManager('face')
# Add cell transition behavior function for all live cells
for cell_id in sheet.face_df.index:
    manager.append(cell_cycle_transition, cell_id=cell_id)
# The History object records all the time steps
history = History(sheet)

manager.update()
solver = EulerSolver(
        sheet, geom, model,
        history=history,
        manager=manager,
        bounds=(-sheet.edge_df.length.mean()/10, sheet.edge_df.length.mean()/10))

solver.solve(tf=5, dt=0.1)
history = solver.history

geom.update_all(sheet)
fig, ax = sheet_view(sheet)
plt.show()




# """test if cell_transition_cycle works alone in event manger."""
# t=0
# stop = 20
# while manager.current and t < stop:
#     manager.execute(sheet)
#     t += 1
#     sheet.reset_index(order=True)
#     S_count = len(sheet.face_df.index[sheet.face_df['cell_class'] == 'S'].tolist())
#     M_count = len(sheet.face_df.index[sheet.face_df['cell_class'] == 'M'].tolist())
#     G1_count = len(sheet.face_df.index[sheet.face_df['cell_class'] == 'G1'].tolist())
#     G2_count = len(sheet.face_df.index[sheet.face_df['cell_class'] == 'G2'].tolist())
#
#     print(f'At time {t:.4f}, there are {S_count} S cells; {G2_count} G2 cells; {M_count} M cells; {G1_count} G1 cells. \n')
#     manager.update()
#
# fig, ax = sheet_view(sheet, mode='2D')
# plt.show()
#


# Apply drawing specs, so STB and CT have different colours.
from tyssue.config.draw import sheet_spec
draw_specs = sheet_spec()
# Enable face visibility.
draw_specs['face']['visible'] = True
for i in sheet.face_df.index:   # Assign face colour based on cell type.
    if sheet.face_df.loc[i,'cell_class'] == 'STB': sheet.face_df.loc[i,'color'] = 0.7
    else: sheet.face_df.loc[i,'color'] = 0.1
draw_specs['face']['color'] = sheet.face_df['color']
draw_specs['face']['alpha'] = 0.2   # Set transparency.

# Set the threshold values for mesh restructure.
t1_threshold = sheet.edge_df['length'].mean()/10
t2_threshold = sheet.face_df['area'].mean()/10
d_min = t1_threshold
d_sep = d_min *1.5
max_movement = t1_threshold / 2
# Start simulating using my own solver.
t = 0
t_end = 1

if Tyssue_Euler_solver == False:
    while t <= t_end:
        dt = 0.001

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
                        # fig, ax = sheet_view(sheet, edge={'head_width': 0.1})
                        # for face, data in sheet.vert_df.iterrows():
                        #     ax.text(data.x, data.y, face)
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
            if cell_fate_roulette <= 0.3: # Use probability of 0.5 for division.
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


        print(f'At time {t:.4f}, there are {S_count} S cells; {G2_count} G2 cells; {M_count} M cells; {G1_count} G1 cells. \n')

        # Generate the plot at this time step.
        # Enable face visibility.
        draw_specs['face']['visible'] = True
        for i in sheet.face_df.index:  # Assign face colour based on cell type.
            if sheet.face_df.loc[i, 'cell_class'] == 'STB':
                sheet.face_df.loc[i, 'color'] = 0.7
            else:
                sheet.face_df.loc[i, 'color'] = 0.1
        draw_specs['face']['color'] = sheet.face_df['color']
        draw_specs['face']['alpha'] = 0.2  # Set transparency.
        fig, ax = sheet_view(sheet, ['x', 'y'], **draw_specs)
        ax.title.set_text(f'time = {round(t, 5)}')
        # Fix axis limits and aspect.
        ax.set_xlim(-5, 20)
        ax.set_ylim(-5, 7)
        ax.set_aspect('equal')
        # Save to file instead of showing.
        frame_path = f"frames/frame_{t:.5f}.png"
        plt.savefig(frame_path)
        plt.close(fig)  # Close figure to prevent memory leaks

        # Update time_point
        t += dt

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
    with imageio.get_writer('bilayer_starts_with_G1.mp4', fps=15, format='ffmpeg') as writer:
        # Read and append each frame in sorted order
        for filename in frame_files:
            image = imageio.imread(filename)  # Load image from the folder
            writer.append_data(image)        # Write image to video



elif Tyssue_Euler_solver:
    manager = EventManager("face", )

    # Implicit Euler solver

    solver = EulerSolver(
        sheet,
        geom,
        model,
        manager=manager,
        bounds=(
            -sheet.edge_df.length.median() / 10,
            sheet.edge_df.length.median() / 10
        )
    )
    manager.update()

    solver.solve(tf=120.0, dt=0.1)

    fig, ax = sheet_view(
        sheet,
        mode="2D",
        face={"visible": True},
        edge={"head_width": 0.0, "color": sheet.edge_df["line_tension"]},
        vert={"visible": False}
    )



print('\n This is the end of this script. (＾• ω •＾) ')
