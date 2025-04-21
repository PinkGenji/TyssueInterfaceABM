"""
This script creates a bilayer geometry, then implement a multi-class cell system to it.

The cell must be an S cell, and must be neighbouring an STB.
We don’t know why cells fuse so we must recruit these cells to fuse (class "F") with some probability pf
"""


# Load all required modules.
import numpy as np
import  re
import matplotlib.pyplot as plt
from tyssue import Sheet
from tyssue.topology.sheet_topology import remove_face
from tyssue import PlanarGeometry as geom #for simple 2d geometry
from tyssue.dynamics import effectors, model_factory

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


rng = np.random.default_rng(70)    # Seed the random number generator.

# Generate the initial cell sheet for bilayer.
num_x = 16
num_y = 4

sheet =Sheet.planar_sheet_2d(identifier='bilayer', nx = num_x, ny = num_y, distx = 1, disty = 1)
geom.update_all(sheet)
#Updates the sheet geometry by updating: * the edge vector coordinates * the edge lengths * the face centroids
# * the normals to each edge associated face * the face areas.

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

# Plot the figure to see the initial setup is what we want.
fig, ax = sheet_view(sheet)
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)
plt.show()
ax.set_title("Initial Bilayer Setup")  # Adding title

print('Initial geometry plot generated. \n')


# Add a new attribute to the face_df, called "cell class"
sheet.face_df['cell_class'] = 'default'
sheet.face_df['timer'] = 'NA'
total_cell_num = len(sheet.face_df)

print('New attributes: cell_class; timer created for all cells. \n ')
for i in range(0,num_x-2):  # These are the indices of bottom layer.
    sheet.face_df.loc[i,'cell_class'] = 'S'

for i in range(num_x-2,len(sheet.face_df)):     # These are the indices of top layer.
    sheet.face_df.loc[i,'cell_class'] = 'STB'

print(f'There are {total_cell_num} total cells; equally split into "S" and "STB" classes. ')

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

# Merge the STB units into one whole STB.
# Repeatedly merge STB–STB neighbors until none remain
while True:
    pair = find_stb_neighbors(sheet)
    if pair is None:
        print("All STB cells are merged.")
        break

    face1, face2 = pair
    if face1 in sheet.face_df.index and face2 in sheet.face_df.index:
        print(f"Merging STB cells {face1} and {face2}")
        cell_merge(sheet, face1, face2, 'STB')

# Finally, update geometry
geom.update_all(sheet)


# Step 3: Update geometry
geom.update_all(sheet)
sheet.get_extra_indices()


# Next, I need to colour STB and others differently, and bold the dummy edges when plotting.
from tyssue.config.draw import sheet_spec
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

fig, ax = sheet_view(sheet, ['x', 'y'], **draw_specs, mode='2D')
plt.show()


# Set the threshold values for mesh restructure.
t1_threshold = sheet.edge_df['length'].mean()/10
t2_threshold = sheet.face_df['area'].mean()/10
d_min = t1_threshold
d_sep = d_min *1.5
max_movement = t1_threshold / 2


# Start simulating.
t = 0
t_end = 1

while t < t_end:
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
        # If it is in neighbour with STB, then use probability to assign into "F" class.
        can_fuse = 0
        neighbours = sheet.get_neighbors(cell)
        for i in neighbours:
            if sheet.face_df.loc[i, 'cell_class'] == 'STB':
                can_fuse = 1
                break
            else:
                continue
        # Use rng to randomly generate a number between 1 and 10, this will determine the fate of the mature CT.
        cell_fate_roulette = rng.random()
        if can_fuse == 1 and cell_fate_roulette < 0.2:  # If CT is adjacent to STB, then it has 20% probability to fuse.
            sheet.face_df.loc[cell, 'cell_class'] = 'F'
            # Add a timer for each cell enters 'F'.
            sheet.face_df.loc[cell, 'timer'] = 0.01
        if can_fuse == 1 and 0.2< cell_fate_roulette <0.5: # If CT is adjacent to STB, it has 30% probability to divide.
            sheet.face_df.loc[cell, 'cell_class'] = 'G2'
            sheet.face_df.loc[cell, 'timer'] = 0.4

        if cell_fate_roulette <= 0.3 and can_fuse==0: # If CT is not adjacent to STB, then divide with probability 50%.
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
            if sheet.face_df.loc[associated_cell, 'cell_class'] == 'STB' and sheet.face_df.loc[
                opposite_cell, 'cell_class'] == 'STB':
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

    # Add trackers for quantify.
    cell_num_count = len(sheet.face_df)
    S_count = len(sheet.face_df.index[sheet.face_df['cell_class'] == 'S'].tolist())
    M_count = len(sheet.face_df.index[sheet.face_df['cell_class'] == 'M'].tolist())
    G1_count = len(sheet.face_df.index[sheet.face_df['cell_class'] == 'G1'].tolist())
    G2_count = len(sheet.face_df.index[sheet.face_df['cell_class'] == 'G2'].tolist())
    F_count = len(sheet.face_df.index[sheet.face_df['cell_class'] == 'F'].tolist())

    print(f'At time {t:.4f}: {F_count} in F; {S_count} in S; {G2_count} in G2; {M_count} in M; {G1_count} in G1. \n')

    # Print the plot at this step.
    fig, ax = sheet_view(sheet, ['x', 'y'], **draw_specs)
    ax.title.set_text(f'time = {round(t, 5)}')
    # Fix axis limits and aspect
    ax.set_xlim(-5, 20)  # Example limits — change to suit your sheet
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    # Save to file instead of showing
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
with imageio.get_writer('bilayer_fusion_as_merge.mp4', fps=15, format='ffmpeg') as writer:
    # Read and append each frame in sorted order
    for filename in frame_files:
        image = imageio.imread(filename)  # Load image from file
        writer.append_data(image)        # Write image to video




print('\n This is the end of this script. (＾• ω •＾) ')
