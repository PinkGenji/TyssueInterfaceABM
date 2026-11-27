
import matplotlib.pyplot as plt
from tyssue import Sheet
from tyssue.draw import sheet_view
from tyssue.io import hdf5
from matplotlib.collections import LineCollection

# ---- HDF5 filenames ----
timepoints = {
    0: ['simulation_0.hdf5', 'simulation_0.hdf5', 'simulation_0.hdf5'],
    6: ['simulation_6.hdf5', 'simulation_6.hdf5', 'simulation_6.hdf5'],
    12: ['simulation_12.hdf5', 'simulation_12.hdf5', 'simulation_12.hdf5'],
    18: ['simulation_18.hdf5', 'simulation_18.hdf5', 'simulation_18.hdf5'],
    'Final': [
        'uniform_alpha_final.hdf5',
        '80percent_alpha_final.hdf5',
        '50percent_alpha_final.hdf5'
    ]
}

# ---- Generate one plot for each timepoint ----
for timepoint, filenames in timepoints.items():

    # ---- Uniform alpha ----
    dsets = hdf5.load_datasets(
        'uniform_alpha_result_plots/' + filenames[0]
    )
    uniform_sheet = Sheet('reloaded', dsets)
    fig, ax = plt.subplots(figsize=(10, 6))

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
    dsets2 = hdf5.load_datasets(
        '80percent_alpha_result_plots/' + filenames[1]
    )
    reduced_sheet = Sheet('reloaded', dsets2)

    STB_ids = range(14, 28)
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
    dsets3 = hdf5.load_datasets(
        '50percent_alpha_result_plots/' + filenames[2]
    )
    half_sheet = Sheet('reloaded', dsets3)

    STB_ids = range(14, 28)
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

    if timepoint == 'Final':
        ax.set_title('STB outer surface comparison - Final')
        filename = 'STB_boundary_Final.png'
    else:
        ax.set_title(f'STB outer surface comparison - {timepoint} hours')
        filename = f'STB_boundary_{timepoint}h.png'

    ax.set_aspect('equal', adjustable='box')
    ax.margins(0.05)
    ax.autoscale_view()
    # ---- Generate legend relative to the figure ----
    handles, labels = ax.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.10, 1, 1])

    plt.show()

    print(f'Generated {filename}')

print('This is the end of the script.')
