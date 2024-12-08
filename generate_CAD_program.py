import Preprocessing.dataloader
import Preprocessing.generate_dataset_baseline
import Preprocessing.gnn_graph
import Preprocessing.gnn_graph_stroke

import Preprocessing.proc_CAD.generate_program
import Preprocessing.proc_CAD.Program_to_STL
import Preprocessing.proc_CAD.brep_read
import Preprocessing.proc_CAD.helper

import whole_process_helper.helper

import Models.loop_embeddings

import Encoders.gnn.gnn
import Encoders.gnn_stroke.gnn
import Encoders.helper

import particle
# import whole_process_evaluate

from torch.utils.data import DataLoader
from tqdm import tqdm

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import os
import shutil
import numpy as np
import random

# --------------------- Dataset --------------------- #
dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/generate_CAD', return_data_path=True)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)


# --------------------- Directory --------------------- #
current_dir = os.getcwd()
output_dir = os.path.join(current_dir, 'program_output')



# --------------------- Main Loop --------------------- #
pass    


# --------------------- Main Code --------------------- #
data_produced = 0
data_limit = 100
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)


for data in tqdm(data_loader, desc="Generating CAD Programs"):
    program, stroke_node_features, data_path= data
    
    if data_produced >= data_limit:
        break

    if program[-1][0] != 'terminate':
        continue
    
    print("program", program)

    # try:
    cur_output_dir = os.path.join(output_dir, f'data_{data_produced}')
    if os.path.exists(cur_output_dir):
        shutil.rmtree(cur_output_dir)
    os.makedirs(cur_output_dir, exist_ok=True)
    

    gt_brep_dir = os.path.join(data_path[0], 'canvas')
    brep_files = [file_name for file_name in os.listdir(gt_brep_dir)
            if file_name.startswith('brep_') and file_name.endswith('.step')]
    brep_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    gt_brep_file_path = os.path.join(gt_brep_dir, brep_files[-1])


    for particle_id in range (50):
        new_particle = particle.Particle(cur_output_dir, gt_brep_file_path, data_produced, stroke_node_features, particle_id)
        while new_particle.is_valid_particle():
            new_particle.generate_next_step()
        
        # if not new_particle.success_terminate:
        #     delete_dir = os.path.join(cur_output_dir, f'particle_{particle_id}')
        #     if os.path.exists(delete_dir):
        #         shutil.rmtree(delete_dir)

    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     if os.path.exists(cur_output_dir):
    #         shutil.rmtree(cur_output_dir)
    #     data_produced -= 1

    data_produced += 1
