"""
This script contains a vertex model that starts with a bilayer structure.
The system has a multi-class system. Three cellular behaviours are modelled: cell proliferation, fusion and extrusion.
This is a baseline model that the only stochasticity is my initial assignment of cell classes.
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

# import my own functions
from my_headers import *

import os
from tyssue.io import hdf5 # For saving the datasets
import imageio.v2 as imageio
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



# Define the directory name
frames_dir = "frames_Usually"
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
# I am using 1 hour = 1 time unit in the simulation, the simulation should run for 4 days. So picking 0<t<96 hours.
tau_G1 = 8   # Min G1 phase time is 8 hours
tau_S = 7    # Min S phase time is 7 hours
tau_G2 = 3   # Min G2 phase time is 3 hours
tau_M = 0.5   # Min M phase time is 0.5 hours
tau_F = 4    # Min F phase time is 24 hours
stb_age = 35      # After 35 hours, STB units can start to extrude with a certain probability at each time step.
print('New attributes: cell_class; timer created for all cells. \n ')

for i in range(0,num_x-2):  # These are the indices of the bottom layer.
    # All CTs assigned with class ‘G1’, ‘S’, ‘M’, or ‘G2’ based on probabilities that reflect typical times in each stage of the cell cycle
    # Draw a random number between 0 and 1, it's G1 if  < 11/24, S if < 19/24, M if < 20/24, else, G2.
    random_num = rng.random()
    if random_num < 11/24:
        sheet.face_df.loc[i,'cell_class'] = 'G1'
        sheet.face_df.loc[i, 'timer'] = tau_G1
    elif 11/24 <= random_num < 19/24:
        sheet.face_df.loc[i,'cell_class'] = 'S'
        sheet.face_df.loc[i, 'timer'] = tau_S
    elif 19/24 <= random_num < 20/24:
        sheet.face_df.loc[i,'cell_class'] = 'M'
        sheet.face_df.loc[i, 'timer'] = tau_M
    else:
        sheet.face_df.loc[i,'cell_class'] = 'G2'
        sheet.face_df.loc[i, 'timer'] = tau_G2

for i in range(num_x-2,len(sheet.face_df)):     # These are the indices of the top layer.
    sheet.face_df.loc[i,'cell_class'] = 'STB'
    sheet.face_df.loc[i, 'timer'] = np.nan

print(f'There are {total_cell_num} total cells; equally split into "G1" and "STB" classes. ')

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
        'contractility': 0,
        'is_alive': 1,
        'prefered_perimeter': 5.3,
        'perimeter_elasticity':110,
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

# Deactivate the edges between STB units.
for i in sheet.edge_df.index:
    if sheet.edge_df.loc[i,'opposite'] != -1:
        associated_cell = sheet.edge_df.loc[i,'face']
        opposite_edge = sheet.edge_df.loc[i,'opposite']
        opposite_cell = sheet.edge_df.loc[opposite_edge,'face']
        if sheet.face_df.loc[associated_cell,'cell_class'] == 'STB' and sheet.face_df.loc[opposite_cell,'cell_class'] == 'STB':
            sheet.edge_df.loc[i,'is_active'] = 0
            sheet.edge_df.loc[opposite_edge,'is_active'] = 0

# Deactivate the vertices associated with the four corner cells
corner_cells = [0, num_x-3, num_x-2, len(sheet.face_df)-1]
fix_vert_set = set()
for cell in corner_cells:
    # Assign these cells into 'boundary' class, hence no cell class change on them.
    sheet.face_df.loc[cell,"cell_class"] = "boundary_fixed"
    sheet.face_df.loc[cell,"timer"] = np.nan
    # sheet.face_df.loc[cell,'is_alive'] = 0
    vert_list = sheet.edge_df[sheet.edge_df['face'] == cell][['srce', 'trgt']].values.flatten()
    fix_vert_set.update(vert_list)
for vert in fix_vert_set:
    sheet.vert_df.loc[vert,'is_active'] = 0

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
t_end = 18

while t <= t_end:
    dt = 0.01  # initial time step, will be updated dynamically later.

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
    boundary_edges, boundary_vertices = boundary_ids(sheet)
    T3(sheet, boundary_vertices, boundary_edges, length_threshold=d_min, multiplier=1.5)
    sheet.reset_index(order=True)
    geom.update_all(sheet)
    sheet.remove(sheet.get_invalid())
    sheet.get_extra_indices()

    ### Cell class governing section starts.
    # For a mature 'S' cell, if it's touching STB, then it's possible to fuse, otherwise, must continue CT cycle.
    S_cells = sheet.face_df.index[sheet.face_df['cell_class'] == 'S'].tolist()
    fusion_count = 0
    for cell in S_cells:
        if sheet.face_df.loc[cell, 'timer'] >0:
            sheet.face_df.loc[cell, 'timer'] -= dt
        else:
            neighbours = list(sheet.get_neighbors(cell))
            # Count the number of STB neighbours the cell has. Only fuse when have at least 2 STB units touching.
            neighbours_df = sheet.face_df.loc[neighbours]
            stb_count = 0 #disable fusion

            # Use rng to randomly generate a number between 0 and 1, this will determine the fate of the mature CT.
            cell_fate_roulette = rng.random()
            if stb_count > 1 and cell_fate_roulette < 0.3:  # If CT is adjacent to STB, then it has 30% probability to fuse.
                sheet.face_df.loc[cell, 'cell_class'] = 'F'
                # Add a timer for each cell enters 'F'.
                sheet.face_df.loc[cell, 'timer'] = tau_F
                fusion_count += 1
            else:   # Otherwise, all the cells becomes a G2 class.
                sheet.face_df.loc[cell, 'cell_class'] = 'G2'
                sheet.face_df.loc[cell, 'timer'] = tau_G2

    # At the end of the timer, "G2" becomes "M".
    G2_cells = sheet.face_df.index[sheet.face_df['cell_class'] == 'G2'].tolist()
    for cell in G2_cells:
        if sheet.face_df.loc[cell, 'timer'] < 0:
            sheet.face_df.loc[cell, 'cell_class'] = 'M'
            sheet.face_df.loc[cell, 'timer'] = tau_M
        else:
            sheet.face_df.loc[cell, 'timer'] -= dt

    # Cell division.
    # For all cells in "M", divide the cell. Then cells become "G1".
    # Cells in "M" class can be divided.
    cells_can_divide = sheet.face_df.index[sheet.face_df['cell_class'] == 'M'].tolist()
    for index in cells_can_divide:
        if sheet.face_df.loc[index, 'timer'] > 0:
            sheet.face_df.loc[index, 'timer'] -= dt
        else:
            daughter_index = division_mt(sheet, rng=rng, cell_id=index)
            sheet.face_df.loc[index, 'cell_class'] = 'G1'
            sheet.face_df.loc[daughter_index, 'cell_class'] = 'G1'
            # Add a timer for each cell enters "G1".
            sheet.face_df.loc[index, 'timer'] = tau_G1
            sheet.face_df.loc[daughter_index, 'timer'] = tau_G1
    sheet.reset_index()
    sheet.reset_topo()
    geom.update_all(sheet)

    # At the end of the timer, "G1" class becomes "S".
    G1_cells = sheet.face_df.index[sheet.face_df['cell_class'] == 'G1'].tolist()
    for cell in G1_cells:
        if sheet.face_df.loc[cell, 'timer'] < 0:
            sheet.face_df.loc[cell, 'cell_class'] = 'S'
            sheet.face_df.loc[cell, 'timer'] = tau_S
        else:
            sheet.face_df.loc[cell, 'timer'] -= dt

    # At the end of a timer, "F" class becomes "STB" and dummy edge is generated.
    F_cells = sheet.face_df.index[sheet.face_df['cell_class'] == 'F'].tolist()
    for cell in F_cells:
        if sheet.face_df.loc[cell, 'timer'] < 0:
            fusing_cell = fuse_single_cell(sheet, cell, 10*d_min)
            fusing_cell_idx = sheet.face_df[sheet.face_df['unique_id'] == fusing_cell].index
            sheet.face_df.loc[fusing_cell_idx, 'cell_class'] = 'STB'
            sheet.face_df.loc[fusing_cell_idx,'timer'] = 0 # As a fresh STB unit, set the timer to be 0
        else:
            sheet.face_df.loc[cell, 'timer'] -= dt
    sheet.reset_index()
    sheet.reset_topo()
    geom.update_all(sheet)

    # # Extrude the 'E' units before assigning new 'E' units.
    # E_units = sheet.face_df.index[sheet.face_df['cell_class'] == 'E'].tolist()
    # for unit in E_units:
    #     stb_extrusion(sheet, unit)
    # geom.update_all(sheet)
    #
    # # Work on STB units
    # STB_units = sheet.face_df.index[sheet.face_df['cell_class'] == 'STB'].tolist()
    # for unit in STB_units:
    #     if sheet.face_df.loc[unit, 'timer'] > stb_age:
    #         sheet.face_df.loc[unit, 'cell_class'] = 'E'  # Mark the cell as extruding.
    #         stb_detach(sheet, geom, unit)
    #     else:
    #         sheet.face_df.loc[unit, 'timer'] += dt
    # geom.update_all(sheet)

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
    real_time_hours = t
    total_STB = sheet.face_df.loc[sheet.face_df['cell_class'] == 'STB', 'area'].sum()
    STB_area.append(total_STB)
    time_list.append(real_time_hours)
    # Record fusion events and time.
    fusion_events.append(fusion_count)
    # Print time in console.
    print(f'At time {real_time_hours:.4f} hours\n')

    # Generate the plot at this time step.
    update_draw_specs(sheet, draw_specs)  # Update drawing specifications based on current sheet state
    fig, ax = sheet_view(sheet, ['x', 'y'], **draw_specs)
    ax.title.set_text(f'time = {real_time_hours:.4f}')
    ax.set_axis_off()
    # Save to file instead of showing.
    frame_path = f"frames_Usually/frame_{real_time_hours:.4f}.png"
    plt.savefig(frame_path)
    plt.close(fig)  # Close figure to prevent memory leaks

    # Update time_point
    t += dt

final_stb_area = sheet.face_df.loc[sheet.face_df['cell_class'] == 'STB', 'area'].sum()
final_stb_ct_interface_length = stb_ct_interface_length(sheet)
final_stb_thickness = final_stb_area/final_stb_ct_interface_length

# Write the final sheet to a hdf5 file.
hdf5.save_datasets('Usual_proliferation.hdf5', sheet)

""" Generate the video based on the frames saved. """
# Path to folder containing the frame images
frame_folder = "frames_Usually"

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
with imageio.get_writer('Usual_proliferation.mp4', fps=15, format='ffmpeg') as writer:
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
plt.savefig("stb_area_over_time.png", dpi=500)
plt.close()



plt.figure(figsize=(8, 5))
plt.plot(time_list, fusion_events, label='Fusion events per time step', color='red')
plt.xlabel('Time')
plt.ylabel('Number of fusion events')
plt.title('Fusion Events Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("fusion_events_over_time.png", dpi=500)
plt.close()

hours_per_step = dt * 100
fusion_rate = [count / hours_per_step for count in fusion_events]

plt.figure(figsize=(8, 5))
plt.plot(time_list, fusion_rate, color='darkred')
plt.xlabel('Time')
plt.ylabel('Fusion events per hour')
plt.title('Fusion Rate Over Time')
plt.grid(True)
plt.tight_layout()
plt.savefig("fusion_rate_over_time.png", dpi=500)
plt.close()


# Create a DataFrame with all tracked quantities
import pandas as pd

df = pd.DataFrame({
    "time": time_list,
    "fusion_events": fusion_events,
    "STB_area": STB_area
})
# Save to CSV
df.to_csv("Usual_proliferation.csv", index=False)
print("Saved csv file \n")

print(f' The initial STB area is {initial_stb_area:.2f},\n the initial STB-CT interface length is {initial_stb_ct_interface_length:.2f},\n and the initial mean thickness is {initial_stb_thickness:.2f}.\n')
print(f' The final STB area is {final_stb_area:.2f},\n the final STB-CT interface length is {final_stb_ct_interface_length:.2f},\n and the final mean thickness is {final_stb_thickness:.2f}.\n')

print('\n This is the end of this script. (＾• ω •＾) ')
