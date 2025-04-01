# tyssueHello
This is the repo to store my vertex modelling code on my project using Tyssue package. 

There are currently two folders: "Hello Tyssue learning" and "Model with multiple cell class".
The first folder is what I used to familiarise myself with Tyssue library, and the second folder contains my ABM that is 
built upon a multi-cell-class system.

The following explains what each file is in the "Model with multiple cell class" folder, and how to run the script if
it is a script that generates a model.

File "**bilayer_time_driven_cell_cycle_model.py**":
This script simulates a time(only)-driven cell cycle using a vertex model with the Tyssue library. Euler simple forward
method is used to update the positions of vertices.
To run it, make sure you have Python installed along with the required dependencies: numpy, matplotlib, and tyssue. 
Then simply execute the script in a Python environment, it will generate an initial cell sheet, simulate cell cycle 
progression, and visualize results. 
The simulation includes cell division and transitions between different cell cycle classes based on predefined 
probabilities and timers from a _bilayer geometry_ initially. 

File "**contact_inhibition_multi_class_model.py**": 
This script simulates a time(only)-driven cell cycle using a vertex model with the Tyssue library. Euler simple forward
method is used to update the positions of vertices.
To run it, make sure you have Python installed along with the required dependencies: numpy, matplotlib, and tyssue. 
Then simply execute the script in a Python environment, it will generate an initial cell sheet, simulate cell cycle 
progression, and visualize results. 
The simulation includes cell division and transitions between different cell cycle classes based on predefined 
probabilities and timers _from a single cell that grows into a petri-dish fashion_. Euler simple forward method is used to update the positions
of vertices.

