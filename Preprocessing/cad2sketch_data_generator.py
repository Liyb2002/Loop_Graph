
import Preprocessing.proc_CAD.proc_gen
import Preprocessing.proc_CAD.CAD_to_stroke_cloud
import Preprocessing.proc_CAD.render_images
import Preprocessing.proc_CAD.Program_to_STL
import Preprocessing.proc_CAD.helper
import Preprocessing.proc_CAD.render_images
import Preprocessing.proc_CAD.draw_all_lines_baseline

import Preprocessing.gnn_graph
import Preprocessing.SBGCN.brep_read

import shutil
import os
import pickle
import torch
import numpy as np
import threading
import re

class cad2sketch_dataset_generator():

    def __init__(self):

        self.generate_dataset()
        print('hey')
    
    def generate_dataset(self):
        pass

