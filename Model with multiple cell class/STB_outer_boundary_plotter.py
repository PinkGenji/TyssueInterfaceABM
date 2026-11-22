
import matplotlib.pyplot as plt
from tyssue import Sheet
from tyssue.draw import sheet_view
from tyssue.io import hdf5
from matplotlib.collections import LineCollection

# ---- Uniform alpha ----
dsets = hdf5.load_datasets('uniform_alpha.hdf5')
uniform_sheet = Sheet('reloaded', dsets)
fig, ax = sheet_view(uniform_sheet)
STB_ids = range(14, 28)
stb_edges = uniform_sheet.edge_df[
    (uniform_sheet.edge_df['face'].isin(STB_ids)) &
    (uniform_sheet.edge_df['opposite'] == -1)
]
segments = [
    [(row.sx, row.sy), (row.tx, row.ty)]
    for row in stb_edges.itertuples()
]
lc = LineCollection(
    segments,
    colors='tab:blue',
    linewidths=1.5,
    label='Uniform area elasticity'
)
ax.add_collection(lc)

# ---- 80% alpha ----
dsets2 = hdf5.load_datasets('80percent_alpha.hdf5')
reduced_sheet = Sheet('reloaded', dsets2)
STB_ids = range(14,28)
stb_edges = reduced_sheet.edge_df[
    (reduced_sheet.edge_df['face'].isin(STB_ids)) &
    (reduced_sheet.edge_df['opposite'] == -1)
]
segments = [
    [(row.sx, row.sy), (row.tx, row.ty)]
    for row in stb_edges.itertuples()
]
lc = LineCollection(
    segments,
    colors='tab:orange',
    linewidths=1.5,
    label='80% area elasticity'
)
ax.add_collection(lc)

# ---- 50% alpha ----
dsets3 = hdf5.load_datasets('50percent_alpha.hdf5')
half_sheet = Sheet('reloaded', dsets3)
STB_ids = range(14,28)
stb_edges = half_sheet.edge_df[
    (half_sheet.edge_df['face'].isin(STB_ids)) &
    (half_sheet.edge_df['opposite'] == -1)
]
segments = [
    [(row.sx, row.sy), (row.tx, row.ty)]
    for row in stb_edges.itertuples()
]
lc = LineCollection(
    segments,
    colors='tab:green',
    linewidths=1.5,
    label='50% area elasticity'
)
ax.add_collection(lc)

# ---- Generate the plot ----
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('STB outer surface comparison')
ax.set_aspect('equal', adjustable='datalim')
ax.autoscale_view()
ax.legend()
plt.tight_layout()
plt.show()

print('This is the end of the script. ')