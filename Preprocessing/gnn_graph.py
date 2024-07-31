import os
import torch

import os
import sys
import torch
import torch_geometric
from torch_geometric.data import Data, HeteroData
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from networkx.algorithms import community
import seaborn as sns

operations_dict = {     "terminate": 0,
                        "sketch": 1,
                        "extrude": 2,
                        "fillet": 3
                    } 

class SketchHeteroData(HeteroData):
    def __init__(self, stroke_loop, brep_loop, stroke_edges, brep_edges, stroke_brep_connect):
        super(SketchHeteroData, self).__init__()

        # Node features
        self['stroke'].x = torch.tensor(stroke_loop, dtype=torch.float)
        self['brep'].z = torch.tensor(brep_loop, dtype=torch.float)

        # Converting adjacency matrices to edge indices
        stroke_edges_indices = torch.nonzero(torch.tensor(stroke_edges, dtype=torch.long))
        brep_edges_indices = torch.nonzero(torch.tensor(brep_edges, dtype=torch.long))
        stroke_brep_connect_indices = torch.nonzero(torch.tensor(stroke_brep_connect, dtype=torch.long))

        # Setting edge indices
        self['stroke', 'connects', 'stroke'].edge_index = stroke_edges_indices.t().contiguous()
        self['brep', 'connects', 'brep'].edge_index = brep_edges_indices.t().contiguous()
        self['stroke', 'connects', 'brep'].edge_index = stroke_brep_connect_indices.t().contiguous()
        self['brep', 'connects', 'stroke'].edge_index = stroke_brep_connect_indices.flip(1).t().contiguous()

