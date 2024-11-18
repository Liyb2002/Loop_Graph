
import Preprocessing.proc_CAD.proc_gen
import Preprocessing.proc_CAD.CAD_to_stroke_cloud
import Preprocessing.proc_CAD.render_images
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
        self.data_path = os.path.join(os.getcwd(), 'dataset', 'cad2sketch')
        self.target_path = os.path.join(os.getcwd(), 'dataset', 'cad2sketch_annotated')
        self.idx = 0


        self.generate_dataset()


    def generate_dataset(self):
        folders = [folder for folder in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, folder))]
        

        for folder in folders:
            # folder_path = 'dataset/cad2sketch/201'
            folder_path = os.path.join(self.data_path, folder)

            subfolders = [sf for sf in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, sf))]
            if not subfolders:
                print(f"  No subfolders found in '{folder}'. Skipping...")
                continue

            for subfolder in subfolders:
                # subfolder_path = 'dataset/cad2sketch/201/51.6_-136.85_1.4'
                subfolder_path = os.path.join(folder_path, subfolder)
                
                self.process_subfolder(folder_path, subfolder_path)
    
    
    def process_subfolder(self, folder_path, subfolder_path):
        json_file_path = os.path.join(subfolder_path, 'final_edges.json')

        if os.path.exists(json_file_path):

            # Create a new folder 'data_{idx}' in the target path
            new_folder_name = f"data_{self.idx}"
            new_folder_path = os.path.join(self.target_path, new_folder_name)
            os.makedirs(new_folder_path, exist_ok=True)
            
            
            # Create '/canvas' and '/shape_info' subdirectories
            canvas_folder = os.path.join(new_folder_path, 'canvas')
            shape_info_folder = os.path.join(new_folder_path, 'shape_info')
            os.makedirs(canvas_folder, exist_ok=True)
            os.makedirs(shape_info_folder, exist_ok=True)

            self.copy_shape_files(folder_path, canvas_folder)


            self.idx += 1


    def copy_shape_files(self, source_path, target_path):
        print('source_path', source_path)
        shape_files = [f for f in os.listdir(source_path) if f.lower().endswith('.stl')]
        
        if not shape_files:
            print("  No .stl files found in the source folder.")
            return
        
        for shape_file in shape_files:
            source_file = os.path.join(source_path, shape_file)
            target_file = os.path.join(target_path, shape_file)
            shutil.copy(source_file, target_file)
            print(f"  Copied {shape_file} to {target_path}")
