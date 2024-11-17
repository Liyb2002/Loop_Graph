
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
        data_path = os.path.join(os.getcwd(), 'dataset', 'cad2sketch')
        target_path = os.path.join(os.getcwd(), 'dataset', 'cad2sketch_annotated')

        self.generate_dataset(data_path)
    

    def generate_dataset(self, data_path):
        folders = [folder for folder in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, folder))]
        

        for folder in folders:
            # folder_path = 'dataset/cad2sketch/201'
            folder_path = os.path.join(data_path, folder)

            subfolders = [sf for sf in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, sf))]
            if not subfolders:
                print(f"  No subfolders found in '{folder}'. Skipping...")
                continue

            for subfolder in subfolders:
                # subfolder_path = 'dataset/cad2sketch/201/51.6_-136.85_1.4'
                subfolder_path = os.path.join(folder_path, subfolder)
                
                self.process_subfolder(subfolder_path)
    
    
    def process_subfolder(self, subfolder_path):
        json_file_path = os.path.join(subfolder_path, 'final_edges.json')

        print(f"  Processing json_file_path: {json_file_path}")
