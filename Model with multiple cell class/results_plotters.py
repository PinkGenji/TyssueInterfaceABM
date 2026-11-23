"""
This is a script that reads simulation results folder that
"""
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from tyssue import Sheet
from tyssue.draw import sheet_view
from tyssue.io import hdf5
from matplotlib.collections import LineCollection

uniform_alpha_csv = pd.read_csv("uniform_alpha_result_plots/uniform_alpha.csv")
half_alpha_csv = pd.read_csv("50percent_alpha_result_plots/50percent_alpha.csv")
reduced_alpha_csv = pd.read_csv("80percent_alpha_result_plots/80percent_alpha.csv")


# Store datasets for comparison
datasets = {
    "uniform alpha": uniform_alpha_csv,
    "reduced alpha": reduced_alpha_csv,
    "half alpha": half_alpha_csv
}

# Define variables to plot
variables = {
    "CT_number": "CT Number",
    "F_number": "F Number",
    "STB_area": "Total STB Area",
    "CT_area": "Total CT Area"
}

# Generate comparison plots
for column, ylabel in variables.items():
    plt.figure(figsize=(9, 5))
    for label, df in datasets.items():
        plt.plot(
            df["time_hours"],
            df[column],
            label=label,
            linewidth=1.5
        )
    plt.xlabel("Time (hours)")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()


print('\n This is the end of the script.')
