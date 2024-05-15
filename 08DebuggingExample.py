# -*- coding: utf-8 -*-
"""
This file is for learning 08DebuggingExample of the tyssue package.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pylab as plt

from tyssue import Sheet, Monolayer, config
from tyssue import SheetGeometry, PlanarGeometry


# What we're here for
from tyssue.topology.sheet_topology import type1_transition
from tyssue.topology.base_topology import collapse_edge, remove_face
from tyssue.topology.sheet_topology import split_vert as sheet_split
from tyssue.topology.bulk_topology import split_vert as bulk_split
from tyssue.topology import condition_4i, condition_4ii

## model and solver
from tyssue.dynamics.sheet_vertex_model import SheetModel as model
from tyssue.solvers.quasistatic import QSSolver
from tyssue.generation import extrude
from tyssue.dynamics import model_factory, effectors















































'''
This is the end of the file.
'''
