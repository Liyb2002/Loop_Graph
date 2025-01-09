import Preprocessing.dataloader
import Preprocessing.generate_dataset_baseline
import Preprocessing.gnn_graph

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
import copy

# --------------------- Dataset --------------------- #
dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/whole', return_data_path=True)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)


# --------------------- Directory --------------------- #
current_dir = os.getcwd()
output_dir = os.path.join(current_dir, 'program_output_dataset')



# --------------------- Main Loop --------------------- #
pass    


# --------------------- Main Code --------------------- #
data_produced = 0
data_limit = 1000
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)


for data in tqdm(data_loader, desc="Generating CAD Programs"):
    program, stroke_node_features, data_path= data
    
    if data_produced >= data_limit:
        break

    if program[-1][0] != 'terminate':
        continue
    

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



    base_particle = particle.Particle(gt_brep_file_path, data_produced, stroke_node_features.squeeze(0).cpu().numpy())
    base_particle.set_gt_program(program)
    particle_list = []
    for particle_id in range (20):
        new_particle = copy.deepcopy(base_particle)
        new_particle.set_particle_id(particle_id, cur_output_dir)
        particle_list.append(new_particle)

    while len(particle_list) > 0:
        # particle.next step 
        for cur_particle in particle_list:
            cur_particle.generate_next_step()

        # resample particles
        particle_list = whole_process_helper.helper.resample_particles(particle_list, cur_output_dir)


    data_produced += 1
