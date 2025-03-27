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
import matplotlib.pyplot as plt
from tyssue import Sheet, config, History
from tyssue import PlanarGeometry as geom #for simple 2d geometry
from tyssue.dynamics import PlanarModel, effectors, model_factory
from tyssue.solvers.viscous import EulerSolver
# 2D plotting
from tyssue.draw import sheet_view
from tyssue.draw.plt_draw import  plot_forces
# import my own functions
import my_headers as mh

rng = np.random.default_rng(70)    # Seed the random number generator.


# Generate the initial cell sheet. Note: 6 horizontal and
num_x = 16
num_y = 4

sheet =Sheet.planar_sheet_2d(identifier='bilayer', nx = num_x, ny = num_y, distx = 1, disty = 1)
geom.update_all(sheet)

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
total_cell_num = len(sheet.face_df)
print('Cell class attribute created for all cells and set value as "default". ')
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

sheet.update_specs(model.specs, reset=True)

sheet.settings['threshold_length'] = 1e-3

sheet.update_specs(config.dynamics.quasistatic_plane_spec())
sheet.face_df["prefered_area"] = sheet.face_df["area"].mean()
history = History(sheet) #, extra_cols={"edge":["dx", "dy", "sx", "sy", "tx", "ty"]})

sheet.vert_df['viscosity'] = 1.0
sheet.edge_df.loc[[0, 17],  'line_tension'] *= 4
sheet.face_df.loc[1,  'prefered_area'] *= 1.2

fig, ax = plot_forces(sheet, geom, model, ['x', 'y'], 1)
plt.show()
history = History(sheet, save_every=2, dt=1)

for i in range(10):
    geom.scale(sheet, 1.02, list('xy'))
    geom.update_all(sheet)
    # record only every `save_every` time
    history.record()


solver = EulerSolver(
    sheet,
    geom,
    model,
    history=history,
    auto_reconnect=True)

def on_topo_change(sheet):
    print('Topology changed!\n')
    print("reseting tension")
    sheet.edge_df["line_tension"] = sheet.specs["edge"]["line_tension"]



res = solver.solve(tf=15, dt=0.05, on_topo_change=on_topo_change,
                   topo_change_args=(solver.eptm,))

create_gif(solver.history, "sheet3.gif", num_frames=120)

Image("sheet3.gif")


# Assigning cells in "S" to "G2" by probability 0.5


# Add a timer for each cell enters "G2".


# At the end of the timer, "G2" becomes "M".


# For all cells in "M", divide the cell with no orientation preference. Then cells becomes "G1".


# Add a timer for each cell enters "G1".


# When timer is end, "G1" class becomes "S".




print('\n This is the end of this script. (＾• ω •＾) ')
