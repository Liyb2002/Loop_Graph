
import Preprocessing.proc_CAD.proc_gen
import Preprocessing.proc_CAD.CAD_to_stroke_cloud
import Preprocessing.proc_CAD.render_images
import Preprocessing.proc_CAD.helper
import Preprocessing.proc_CAD.render_images
import Preprocessing.proc_CAD.draw_all_lines_baseline

import Preprocessing.gnn_graph
import Preprocessing.SBGCN.brep_read

import Preprocessing.proc_CAD.cad2sketch_stroke_features

from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCP.StlAPI import StlAPI_Reader
from OCP.TopoDS import TopoDS_Shape
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.IFSelect import IFSelect_RetDone

import json
import shutil
import os
import pickle
import torch
import numpy as np
import threading
import re
import trimesh



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
        strokes_dict_path = os.path.join(subfolder_path, 'strokes_dict.json')

        if not os.path.exists(json_file_path):
            return        
    
        # Create a new folder 'data_{idx}' in the target path
        new_folder_name = f"data_{self.idx}"
        new_folder_path = os.path.join(self.target_path, new_folder_name)
        os.makedirs(new_folder_path, exist_ok=True)
        
        
        # Create '/canvas' and '/shape_info' subdirectories
        canvas_folder = os.path.join(new_folder_path, 'canvas')
        shape_info_folder = os.path.join(new_folder_path, 'shape_info')
        os.makedirs(canvas_folder, exist_ok=True)
        os.makedirs(shape_info_folder, exist_ok=True)

        # self.copy_shape_files(folder_path, canvas_folder)
        self.idx += 1

        # Node connection_matrix
        strokes_dict_data = self.read_json(strokes_dict_path)
        self.compute_connection_matrix(strokes_dict_data)

        # Node Features
        json_data = self.read_json(json_file_path)
        Preprocessing.proc_CAD.cad2sketch_stroke_features.vis_stroke_cloud(json_data)
        self.compute_shape_info(json_data)



    def compute_connection_matrix(self, json_data):
        # Extract all unique IDs
        ids = [d['id'] for d in json_data]
        id_to_index = {id_: index for index, id_ in enumerate(ids)}

        # Initialize the matrix with zeros
        matrix_size = len(ids)
        connection_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

        # Populate the connection matrix
        for dict_item in json_data:
            id_ = dict_item['id']
            intersections = dict_item['intersections']
            id_index = id_to_index[id_]
            
            # If intersections is a list of lists, flatten it
            for sublist in intersections:
                if isinstance(sublist, list):
                    for sub_id in sublist:
                        if sub_id in id_to_index:
                            sublist_index = id_to_index[sub_id]
                            connection_matrix[id_index][sublist_index] = 1
                            connection_matrix[sublist_index][id_index] = 1  # For undirected connection
                elif sublist in id_to_index:
                    sublist_index = id_to_index[sublist]
                    connection_matrix[id_index][sublist_index] = 1
                    connection_matrix[sublist_index][id_index] = 1  # For undirected connection

        return connection_matrix


    def compute_shape_info(self, json_data):
        stroke_node_features = Preprocessing.proc_CAD.cad2sketch_stroke_features.build_final_edges_json(json_data)
        # stroke_operations_order_matrix = ??


    def copy_shape_files(self, source_path, target_path):
        """
        Copies all '.obj' files from the source folder to the target path, converts them to '.stl', 
        and then converts them to '.step'. Keeps both '.stl' and '.step' files in the new folder.
        """
        shape_files = [f for f in os.listdir(source_path) if f.lower().endswith('.obj')]

        if not shape_files:
            print("  No .obj files found in the source folder.")
            return

        for obj_file in shape_files:
            source_file = os.path.join(source_path, obj_file)

            # Copy the .obj file to the target folder
            shutil.copy(source_file, target_path)
            print(f"Copied {obj_file} to {target_path}")

            # Convert the .obj file to .stl
            stl_file_name = os.path.splitext(obj_file)[0] + ".stl"
            stl_file_path = os.path.join(target_path, stl_file_name)
            if self.convert_obj_to_stl(source_file, stl_file_path):
                print(f"Converted {obj_file} to {stl_file_name}")
            else:
                print(f"Failed to convert {obj_file} to .stl")
                continue

            # Convert the .stl file to .step
            step_file_name = os.path.splitext(obj_file)[0] + ".step"
            step_file_path = os.path.join(target_path, step_file_name)
            if self.convert_stl_to_step(stl_file_path, step_file_path):
                print(f"Converted {stl_file_name} to {step_file_name}")
            else:
                print(f"Failed to convert {stl_file_name} to .step")


    def convert_obj_to_stl(self, obj_file, stl_file):
        """
        Converts an .obj file to .stl using trimesh.
        """
        try:
            # Load the OBJ file
            mesh = trimesh.load(obj_file, force='mesh')
            if not isinstance(mesh, trimesh.Trimesh):
                print(f"No valid mesh found in {obj_file}")
                return False

            # Export the mesh to STL
            mesh.export(stl_file, file_type='stl')
            return True
        except Exception as e:
            print(f"Error converting OBJ to STL: {e}")
            return False


    def convert_stl_to_step(self, stl_file, step_file):
        """
        Converts an .stl file to .step using Open CASCADE.
        """
        try:
            # Read the STL file using Open CASCADE
            stl_reader = StlAPI_Reader()
            shape = TopoDS_Shape()
            if not stl_reader.Read(shape, stl_file):
                print(f"Error reading STL file: {stl_file}")
                return False
            
            # Perform meshing (optional but recommended)
            BRepMesh_IncrementalMesh(shape, 0.1)

            # Write to STEP file
            step_writer = STEPControl_Writer()
            step_writer.Transfer(shape, STEPControl_AsIs)
            status = step_writer.Write(step_file)
            
            return status == IFSelect_RetDone
        except Exception as e:
            print(f"Error converting STL to STEP: {e}")
            return False


    def read_json(self, file_path):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            return data
        except Exception as e:
            print(f"Error reading JSON file {file_path}: {e}")
            return None


