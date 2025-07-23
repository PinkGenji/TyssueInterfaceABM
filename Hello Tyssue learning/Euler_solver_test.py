
# Load all required modules.
import random
import numpy as np
import matplotlib.pyplot as plt
from tyssue import Sheet, PlanarGeometry, History
from tyssue.dynamics import effectors, model_factory
from tyssue.topology.sheet_topology import cell_division
from tyssue.behaviors import EventManager
from tyssue.solvers.viscous import EulerSolver

# 2D plotting
from tyssue.draw import sheet_view


random.seed(42)  # Controls Python's random module (e.g. event shuffling)
np.random.seed(42) # Controls NumPy's RNG (e.g. vertex positions, topology)
rng = np.random.default_rng(70)    # Seed the random number generator for my own division function.

# Generate the initial cell sheet for bilayer.
geom = PlanarGeometry
print('\n Initialising the geometry...')
num_x = 16
num_y = 4

sheet = Sheet.planar_sheet_2d(identifier='demo', nx = num_x, ny = num_y, distx = 1, disty = 1)
geom.update_all(sheet)

# remove non-enclosed faces
sheet.remove(sheet.get_invalid())
sheet.sanitize()
sheet.reset_index()
geom.update_all(sheet)

# Plot the figure to see the initial setup is what we want.
fig, ax = sheet_view(sheet)
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)
plt.show()
ax.set_title("Initial Geometry Setup")  # Adding title

print('Initial geometry plot is generated. \n')

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

def cell_cycle_transition(sheet, manager, cell_id=0, p_recruit=0.3, dt=0.1, G2_duration=0.4, G1_duration=0.11):
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
    print(f'checking cell {cell_id}...')
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
        #daughter = division_mt_ver2(sheet, rng=np.random.default_rng(), cent_data=centre_data, cell_id=cell_id)
        # Set parent and daughter to G1 with G1 timer
        sheet.face_df.loc[cell_id, 'cell_class'] = 'G1'
        #sheet.face_df.loc[daughter, 'cell_class'] = 'G1'
        sheet.face_df.loc[cell_id, 'timer'] = G1_duration
        #sheet.face_df.loc[daughter, 'timer'] = G1_duration
        # update topology
        #sheet.reset_index(order = False)
        # update geometry
        #geom.update_all(sheet)
        # append to next deque
        #manager.append(cell_cycle_transition, cell_id = daughter)
        manager.append(cell_cycle_transition, cell_id = cell_id)

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

solver = EulerSolver(
        sheet, geom, model,
        history=history,
        manager=manager,
        bounds=(-sheet.edge_df.length.mean()/10, sheet.edge_df.length.mean()/10))

solver.solve(tf=1, dt=0.1)
history = solver.history

fig, ax = sheet_view(sheet)
plt.show()


print('\n This is the end of this script. (＾• ω •＾) ')
