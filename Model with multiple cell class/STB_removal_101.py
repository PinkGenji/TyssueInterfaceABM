"""
This script creates a trilayer geometry, then artificially detach an STB unit from the CTs.
The goal of this script is to investigate how we should remove an STB unit from its layer.
"""


# Load all required modules.
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from tyssue import Sheet
from tyssue.topology.sheet_topology import remove_face
from tyssue import PlanarGeometry as geom #for simple 2d geometry
from tyssue.dynamics import effectors, model_factory
from tyssue.io import hdf5 # For saving the datasets

# 2D plotting
from tyssue.draw import sheet_view
from tyssue.topology.sheet_topology import cell_division
from tyssue.config.draw import sheet_spec
from tyssue.draw.plt_draw import plot_forces

# Surpress the warnings
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)


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


# Initialise the geometry first.

rng = np.random.default_rng(70)    # Seed the random number generator.

# Generate the initial cell sheet for tri-layer.
num_x = 16
num_y = 5

sheet =Sheet.planar_sheet_2d(identifier='bilayer', nx = num_x, ny = num_y, distx = 1, disty = 1)
geom.update_all(sheet)

# remove non-enclosed faces
sheet.remove(sheet.get_invalid())

# Delete the irregular polygons.
for i in sheet.face_df.index:
    if sheet.face_df.loc[i,'num_sides'] != 6:
        delete_face(sheet,i)
    else:
        continue

sheet.reset_index(order=True)   #continuous indices in all df, vertices clockwise
geom.update_all(sheet)
sheet.get_opposite()
# Add a new attribute to the face_df, called "cell class"
sheet.face_df['cell_class'] = 'default'
sheet.face_df['timer'] = np.nan

for i in range(0,2*num_x-4):  # Looping over the bottom layer.
    sheet.face_df.loc[i,'cell_class'] = 'S'

for i in range(2*num_x-4,len(sheet.face_df)):     # These are the indices of the top layer.
    sheet.face_df.loc[i,'cell_class'] = 'STB'


# Add dynamics to the model, so we can draw based on edge status.
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


# Deactivate the edges between STB units, that is, becoming dummy edges.
for i in sheet.edge_df.index:
    if sheet.edge_df.loc[i,'opposite'] != -1:
        associated_cell = sheet.edge_df.loc[i,'face']
        opposite_edge = sheet.edge_df.loc[i,'opposite']
        opposite_cell = sheet.edge_df.loc[opposite_edge,'face']
        if sheet.face_df.loc[associated_cell,'cell_class'] == 'STB' and sheet.face_df.loc[opposite_cell,'cell_class'] == 'STB':
            sheet.edge_df.loc[i,'is_active'] = 0
            sheet.edge_df.loc[opposite_edge,'is_active'] = 0

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
plt.show()
print('Initialised the geometry.')
# Create force plot
fig, ax = plot_forces(sheet, geom, model, ['x', 'y'], scaling=0.01)
plt.show()

# Detach the STB unit number 34 from the two cells beneath it.

sheet.get_extra_indices()
# find the edge number by checking the mutual edge between cell 19 and 34.
for i in sheet.edge_df.index:
    if sheet.edge_df.loc[i, 'face'] == 19:
        opposite_edge = sheet.edge_df.loc[i, 'opposite']
        if opposite_edge != -1 and sheet.edge_df.loc[opposite_edge, 'face'] == 34:
            edge_to_process = i
            print(f'Edge {edge_to_process} is the edge we want to do T1 transition.')
            # Do a T1 transition on the edge we want, find the edge number first.
            type1_transition(sheet, edge_to_process, remove_tri_faces=False, multiplier=5)
            geom.update_all(sheet)
            sheet.reset_index(order=True)
    else:
        continue

# find the edge number by checking the mutual edge between cell 20 and 34.
sheet.get_extra_indices()
for i in sheet.edge_df.index:
    if sheet.edge_df.loc[i, 'face'] == 20:
        opposite_edge = sheet.edge_df.loc[i, 'opposite']
        if opposite_edge != -1 and sheet.edge_df.loc[opposite_edge, 'face'] == 34:
            edge_to_process = i
            print(f'Edge {edge_to_process} is the edge we want to do T1 transition.')
            # Do a T1 transition on the edge we want, find the edge number first.
            type1_transition(sheet, edge_to_process, remove_tri_faces=False, multiplier=5)
            geom.update_all(sheet)
            sheet.reset_index(order=True)
    else:
        continue


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
plt.show()
print('STB unit 34 is detached.')
# Create force plot
fig, ax = plot_forces(sheet, geom, model, ['x', 'y'], scaling=0.01)
plt.show()

# Next, extrude the STB unit number 34, then check if there are any unwanted vertices.
extrude_face(sheet, 34)
geom.update_all(sheet)
# Create the force plot.
fig, ax = plot_forces(sheet, geom, model, ['x', 'y'], scaling=0.01)
plt.show()


# Assign the dummy edges if they are the mutual edges between STB units.
for i in sheet.edge_df.index:
    if sheet.edge_df.loc[i,'opposite'] != -1:
        associated_cell = sheet.edge_df.loc[i,'face']
        opposite_edge = sheet.edge_df.loc[i,'opposite']
        opposite_cell = sheet.edge_df.loc[opposite_edge,'face']
        if sheet.face_df.loc[associated_cell,'cell_class'] == 'STB' and sheet.face_df.loc[opposite_cell,'cell_class'] == 'STB':
            sheet.edge_df.loc[i,'is_active'] = 0
            sheet.edge_df.loc[opposite_edge,'is_active'] = 0
        else:
            sheet.edge_df.loc[i,'is_active'] = 1
            sheet.edge_df.loc[opposite_edge,'is_active'] = 1
    else:
        sheet.edge_df.loc[i,'is_active'] = 1

# Update face draw specs
draw_specs['face']['visible'] = True
for i in sheet.face_df.index:   # Assign face colour based on cell type.
    if sheet.face_df.loc[i,'cell_class'] == 'STB': sheet.face_df.loc[i,'color'] = 0.7
    else: sheet.face_df.loc[i,'color'] = 0.1
draw_specs['face']['color'] = sheet.face_df['color']
draw_specs['face']['alpha'] = 0.2   # Set transparency.

# update the edge specs
draw_specs['edge']['visible'] = True
for i in sheet.edge_df.index:
    if sheet.edge_df.loc[i,'is_active'] == 0: sheet.edge_df.loc[i,'width'] = 2
    else: sheet.edge_df.loc[i,'width'] = 0.5
draw_specs['edge']['width'] = sheet.edge_df['width']

fig, ax = sheet_view(sheet, ['x', 'y'], **draw_specs)
plt.show()

fig, ax = sheet_view(sheet, ['x', 'y'], **draw_specs)
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face, fontsize=10, color="r")
plt.show()

# Use QS solver to start with steady state
from tyssue.dynamics.planar_vertex_model import PlanarModel as smodel
from tyssue.solvers import QSSolver

solver = QSSolver()
res = solver.find_energy_min(sheet, geom, smodel)
print("Successfull gradient descent? ", res['success'])

# Update face draw specs
draw_specs['face']['visible'] = True
for i in sheet.face_df.index:   # Assign face colour based on cell type.
    if sheet.face_df.loc[i,'cell_class'] == 'STB': sheet.face_df.loc[i,'color'] = 0.7
    else: sheet.face_df.loc[i,'color'] = 0.1
draw_specs['face']['color'] = sheet.face_df['color']
draw_specs['face']['alpha'] = 0.2   # Set transparency.

# update the edge specs
draw_specs['edge']['visible'] = True
for i in sheet.edge_df.index:
    if sheet.edge_df.loc[i,'is_active'] == 0: sheet.edge_df.loc[i,'width'] = 2
    else: sheet.edge_df.loc[i,'width'] = 0.5
draw_specs['edge']['width'] = sheet.edge_df['width']

fig, ax = sheet_view(sheet, ['x', 'y'], **draw_specs)
plt.show()




# Next, let the system evolve and generate a video. Set the threshold values for mesh restructure.
t1_threshold = sheet.edge_df['length'].mean()/10
t2_threshold = sheet.face_df['area'].mean()/10

d_min = t1_threshold
d_sep = d_min *1.5
max_movement = t1_threshold / 2

# Start simulating.
t = 0
t_end = 3

tracked_faces = [28, 32, 33, 34, 35]
energy_log = []  # This will be a list of dicts, one per time step

time_array=[]
area28= []
area32=[]
area33=[]
area34=[]
area35=[]


while t < t_end:
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
                # Process identified edge with T1 transition
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

    # # T3 transition.
    # while True:
    #     T3_todo = None
    #     # print('computing boundary indices.')
    #     boundary_vert, boundary_edge = find_boundary(sheet)
    #
    #     for edge_e in boundary_edge:
    #         # Extract source and target vertex IDs
    #         srce_id, trgt_id = sheet.edge_df.loc[edge_e, ['srce', 'trgt']]
    #         for vertex_v in boundary_vert:
    #             if vertex_v == srce_id or vertex_v == trgt_id:
    #                 continue
    #
    #             distance, nearest = dist_computer(sheet, edge_e, vertex_v, d_sep)
    #             if distance < d_min:
    #                 T3_todo = vertex_v
    #                 # print(f'Found incoming vertex: {vertex_v} and colliding edge: {edge_e}')
    #                 T3_swap(sheet, edge_e, vertex_v, nearest, d_sep)
    #                 sheet.reset_index(order=False)
    #                 geom.update_all(sheet)
    #                 sheet.get_extra_indices()
    #                 # fig, ax = sheet_view(sheet, edge={'head_width': 0.1})
    #                 # for face, data in sheet.vert_df.iterrows():
    #                 #     ax.text(data.x, data.y, face)
    #                 break
    #
    #         if T3_todo is not None:
    #             break  # Exit outer loop to restart with updated boundary
    #
    #     if T3_todo is None:
    #         break

    # Select all mature "S" cells.
    S_cells = sheet.face_df.index[sheet.face_df['cell_class'] == 'S'].tolist()

    for cell in S_cells:
        # Use rng to randomly generate a number between 1 and 10, this will determine the fate of the mature CT.
        cell_fate_roulette = rng.random()

        if cell_fate_roulette <= 0.3: # If CT is not adjacent to STB, then divide with probability 50%.
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

    # At the end of the timer, "G1" class becomes "S".
    G1_cells = sheet.face_df.index[sheet.face_df['cell_class'] == 'G1'].tolist()
    for cell in G1_cells:
        if sheet.face_df.loc[cell, 'timer'] < 0:
            sheet.face_df.loc[cell, 'cell_class'] = 'S'
        else:
            sheet.face_df.loc[cell, 'timer'] -= dt

    sheet.reset_index(order=True)
    geom.update_all(sheet)

    # At the end of the timer, "F" class becomes "STB.
    F_cells = sheet.face_df.index[sheet.face_df['cell_class'] == 'F'].tolist()
    for cell in F_cells:
        if sheet.face_df.loc[cell, 'timer'] < 0:
            sheet.face_df.loc[cell, 'cell_class'] = 'STB'
        else:
            sheet.face_df.loc[cell, 'timer'] -= dt

    geom.update_all(sheet)

    # Before computation the force, we need to make sure we disable the correct dummy edges.
    sheet.get_extra_indices()  # make sure we have correct opposite edges computed.
    for i in sheet.edge_df.index:
        # For a non-boundary edge, if both of itself and its opposite edge are STB class, disable it. Otherwise, make it active.
        if sheet.edge_df.loc[i, 'opposite'] != -1:
            associated_cell = sheet.edge_df.loc[i, 'face']
            opposite_edge = sheet.edge_df.loc[i, 'opposite']
            opposite_cell = sheet.edge_df.loc[opposite_edge, 'face']
            if sheet.face_df.loc[associated_cell, 'cell_class'] == 'STB' and sheet.face_df.loc[opposite_cell, 'cell_class'] == 'STB':
                sheet.edge_df.loc[i, 'is_active'] = 0
                sheet.edge_df.loc[opposite_edge, 'is_active'] = 0
            else:
                sheet.edge_df.loc[i, 'is_active'] = 1
                sheet.edge_df.loc[opposite_edge, 'is_active'] = 1
        # Boundary edges are always active in this model.
        else:
            sheet.edge_df.loc[i, 'is_active'] = 1

    # And update the drawing specs correctly according to active or not (dummy edge is bold).
    # Assign cell colour by cell type. Pale yellow for STB, light purple for CTs.
    for i in sheet.face_df.index:
        if sheet.face_df.loc[i, 'cell_class'] == 'STB':
            sheet.face_df.loc[i, 'color'] = 0.7
        else:
            sheet.face_df.loc[i, 'color'] = 0.1
    draw_specs['face']['color'] = sheet.face_df['color']
    # Assign edge thickness by its type.
    for i in sheet.edge_df.index:
        if sheet.edge_df.loc[i, 'is_active'] == 0:
            sheet.edge_df.loc[i, 'width'] = 2
        else:
            sheet.edge_df.loc[i, 'width'] = 0.5
    draw_specs['edge']['width'] = sheet.edge_df['width']

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


    # Then reload the draw sepcs
    draw_specs['face']['visible'] = True
    for i in sheet.face_df.index:  # Assign face colour based on the cell type.
        if sheet.face_df.loc[i, 'cell_class'] == 'STB':
            sheet.face_df.loc[i, 'color'] = 0.7
        else:
            sheet.face_df.loc[i, 'color'] = 0.1
    draw_specs['face']['color'] = sheet.face_df['color']
    draw_specs['face']['alpha'] = 0.2  # Set transparency.

    draw_specs['edge']['visible'] = True
    for i in sheet.edge_df.index:
        if sheet.edge_df.loc[i, 'is_active'] == 0:
            sheet.edge_df.loc[i, 'width'] = 2
        else:
            sheet.edge_df.loc[i, 'width'] = 0.5
    draw_specs['edge']['width'] = sheet.edge_df['width']

    print(f'Time at {t:.4f} computed.')
    # Generate the plot at this time step.
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

    # Record the energy terms for cell 28, 32, 33, 34 and 35 to get ready for plot over time.
    energy_data = record_cell_energy_dynamic(sheet, model, tracked_faces)
    energy_log.append(energy_data)
    area28.append(sheet.face_df.loc[28,'area'])
    area32.append(sheet.face_df.loc[32, 'area'])
    area33.append(sheet.face_df.loc[33, 'area'])
    area34.append(sheet.face_df.loc[34, 'area'])
    area35.append(sheet.face_df.loc[35, 'area'])
    time_array.append(float(t))
    # Update time_point
    t += dt

# Convert to long-form DataFrame dynamically using model.labels
long_data = []

for t, snapshot in enumerate(energy_log):
    for face_id, energies in snapshot.items():
        row = {'time': t, 'face': face_id}
        row.update(energies)  # energies keys are from model.labels
        long_data.append(row)


df = pd.DataFrame(long_data)
for label in model.labels:
    plt.figure()
    for face_id in df['face'].unique():
        sub_df = df[df['face'] == face_id]
        plt.plot(sub_df['time'], sub_df[label], label=f'Face {face_id}')
    plt.title(f'{label} over time')
    plt.xlabel('Time step')
    plt.ylabel(f'{label} energy')
    plt.legend(title='Face ID')
    plt.tight_layout()
    plt.show()

# Produce a single plot that has 5 curves, one for each face.
plt.figure(figsize=(8, 6))
plt.plot(time_array, area28, label='Face 28')
plt.plot(time_array, area32, label='Face 32')
plt.plot(time_array, area33, label='Face 33')
plt.plot(time_array, area34, label='Face 34')
plt.plot(time_array, area35, label='Face 35')

plt.title('Area of Faces Over Time')
plt.xlabel('Time')
plt.ylabel('Area')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save energy_log into CSV for later analysis
df.to_csv('New_T1_test.csv', index=False)





# Write the final sheet to a hdf5 file.
hdf5.save_datasets('New_T1_test.hdf5', sheet)

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
with imageio.get_writer('New_T1_test.mp4', fps=15, format='ffmpeg') as writer:
    # Read and append each frame in sorted order
    for filename in frame_files:
        image = imageio.imread(filename)  # Load image from the folder
        writer.append_data(image)        # Write image to video



print('\n This is the end of this script. (＾• ω •＾) ')
