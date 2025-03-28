"""
This script models a time driven (and only time) cell cycle. Following are the aims:
    (1) Mature CTs Cell gets recruited from mature into G2 with simple probability p.
    (2) Cell does not grow but is programmed to stay in G2 for a fixed period of time.
    (3) Cell moves to M phase and divides.
    (4) Two daughter cells now in G1 don’t actively grow, but instead are programmed to stay in G1 for fixed amount of time.
    (5) Cells move from G1 to S.
    Note: the only 'cell growth' factor is the target area term in the energy function.
"""


# Load all required modules.
import numpy as np
from decimal import Decimal, getcontext
import matplotlib.pyplot as plt
from tyssue import Sheet, config, History
from tyssue import PlanarGeometry as geom #for simple 2d geometry
from tyssue.dynamics import PlanarModel, effectors, model_factory
from tyssue.topology.sheet_topology import  type1_transition
# 2D plotting
from tyssue.draw import sheet_view
from tyssue.topology.sheet_topology import cell_division

# import my own functions
import my_headers as mh

rng = np.random.default_rng(70)    # Seed the random number generator.


# Generate the initial cell sheet. Note: 6 horizontal and
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
        mh.delete_face(sheet,i)
    else:
        continue

sheet.reset_index(order=True)   #continuous indices in all df, vertices clockwise
geom.update_all(sheet)

# Plot the figure to see the initial setup is what we want.
fig, ax = sheet_view(sheet)
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)
plt.show()
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
model = model_factory([
    effectors.LineTension,
    effectors.FaceContractility,
    effectors.FaceAreaElasticity
])
sheet.vert_df['viscosity'] = 1.0

sheet.update_specs(model.specs, reset=True)

# Set up the numerical calculation.
t_0 = Decimal('0.0')
t_end = Decimal('3.0')
dt = Decimal('0.001')
t = t_0

cell1_class = sheet.face_df.loc[1,'cell_class']
print(f'Cell 1 is in class: "{cell1_class}" at t=0.')

while t <= t_end:
    S_cells = sheet.face_df.index[sheet.face_df['cell_class'] == 'S'].tolist()

    for cell in S_cells:
        # Use rng to randomly generate a number between 1 and 10, this will determine the fate of the mature CT.
        cell_fate_roulette = rng.random()
        if cell_fate_roulette <= 0.1:
            sheet.face_df.loc[cell, 'cell_class'] = 'G2'
            if cell == 1:
                print(f'Cell 1 enter "G2" at time {t}. ')
            # Add a timer for each cell enters "G2".
            sheet.face_df.loc[cell, 'timer'] = Decimal('0.3')
        else:
            continue
    geom.update_all(sheet)

    # At the end of the timer, "G2" becomes "M".
    G2_cells = sheet.face_df.index[sheet.face_df['cell_class'] == 'G2'].tolist()
    for cell in G2_cells:
        if sheet.face_df.loc[cell, 'timer'] == 0:
            sheet.face_df.loc[cell, 'cell_class'] = 'M'
            if cell == 1:
                print(f'Cell 1 enter "M" at time {t}. ')
        else:
            sheet.face_df.loc[cell, 'timer'] -= dt

# For all cells in "M", divide the cell with no orientation preference. Then cells becomes "G1".
    M_cells = sheet.face_df.index[sheet.face_df['cell_class'] == 'M'].tolist()
    for cell in M_cells:
        daugther = cell_division(sheet, cell, geom)
        sheet.face_df.loc[cell, 'cell_class'] = 'G1'
        sheet.face_df.loc[daugther, 'cell_class'] = 'G1'
        if cell == 1:
            print(f'Cell 1 enter "G1" at time {t}. ')
        # Add a timer for each cell enters "G1".
        sheet.face_df.loc[cell, 'timer'] = Decimal('0.3')
        sheet.face_df.loc[daugther, 'timer'] = Decimal('0.3')

    geom.update_all(sheet)

    # At the end of the timer, "G1" class becomes "S".
    G1_cells = sheet.face_df.index[sheet.face_df['cell_class'] == 'G1'].tolist()
    for cell in G1_cells:
        if sheet.face_df.loc[cell, 'timer'] == 0:
            sheet.face_df.loc[cell, 'cell_class'] = 'S'
            if cell == 1:
                print(f'Cell 1 enter "S" at time {t}. ')
        else:
            sheet.face_df.loc[cell, 'timer'] -= dt

    # # Force computing and updating positions.
    # valid_active_verts = sheet.active_verts[sheet.active_verts.isin(sheet.vert_df.index)]
    # pos = sheet.vert_df.loc[valid_active_verts, sheet.coords].values
    # # Compute the moving direction.
    # dot_r = mh.my_ode(sheet)
    # dot_r = Decimal(dot_r)
    # new_pos = pos + dot_r * dt
    # # Save the new positions back to `vert_df`
    # sheet.vert_df.loc[valid_active_verts, sheet.coords] = new_pos
    # geom.update_all(sheet)
    #
    geom.update_all(sheet)
    t += dt

geom.update_all(sheet)
fig, ax = sheet_view(sheet)
plt.show()


print('\n This is the end of this script. (＾• ω •＾) ')
