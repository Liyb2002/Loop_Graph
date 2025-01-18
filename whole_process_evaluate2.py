
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from tqdm import tqdm
import torch
import re


import Preprocessing.SBGCN.brep_read
import Preprocessing.proc_CAD.helper
import Encoders.helper

import fidelity_score

# --------------------- Dataloader for output --------------------- #
class Evaluation_Dataset(Dataset):
    def __init__(self, dataset):
        self.data_path = os.path.join(os.getcwd(), dataset)
        self.data_particles = []
        self.data_pieces = []

        self.data_dirs = [
            os.path.join(self.data_path, d) 
            for d in os.listdir(self.data_path) 
            if os.path.isdir(os.path.join(self.data_path, d))
        ]

        for data_dir in self.data_dirs:
            found_folder = False
            for subfolder in os.listdir(data_dir):
                # Check if the item is a directory and ends with '_output'
                if os.path.isdir(os.path.join(data_dir, subfolder)) and subfolder.endswith('_output'):
                    self.data_particles.append([os.path.join(data_dir, subfolder)])
                    found_folder = True
            if not found_folder:
                self.data_particles.append([])



        # all flatter all the particles
        self.flatted_particle_folders = [
            folder
            for sublist in self.data_particles
            for folder in sublist
        ]

        print("self.flatted_particle_folders", self.data_particles)


        for folder in self.flatted_particle_folders:
            gt_brep_path = os.path.join(folder, 'gt_brep.step')
            canvas_dir = os.path.join(folder, 'canvas')
            
            if os.path.exists(canvas_dir) and os.path.isdir(canvas_dir):
                # Find all files matching the pattern 'brep_{num}.step'
                files = os.listdir(canvas_dir)
                brep_files = [f for f in files if re.match(r'brep_\d+\.step$', f)]
                
                # Extract the highest number from the filenames
                if brep_files:
                    brep_numbers = [int(re.search(r'brep_(\d+)\.step$', f).group(1)) for f in brep_files]
                    highest_num = max(brep_numbers)
                    highest_brep_file = f'brep_{highest_num}.step'
                    highest_brep_path = os.path.join(canvas_dir, highest_brep_file)
                    
                    # Append ground truth and highest brep file paths
                    self.data_pieces.append((gt_brep_path, highest_brep_path))

        print(f"Total number of data pieces: {len(self.data_pieces)}")

    def __len__(self):
        return len(self.data_pieces)

    def __getitem__(self, idx):
        gt_brep_path, output_brep_path = self.data_pieces[idx]
        return gt_brep_path, output_brep_path




# --------------------- Main Code --------------------- #


def run_eval():
    # Set up dataloader
    dataset = Evaluation_Dataset('program_output_test')
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    total_correct = 0
    total = 0

    for data in tqdm(data_loader, desc="Evaluating CAD Programs"):
        gt_brep_path, output_brep_path = data

        print("gt_brep_path", gt_brep_path)
        print("output_brep_path", output_brep_path)

        total += 1
    print(f"Overall Average Accuracy: {total_correct / total:.4f}, with total_correct : {total_correct} and total: {total}")


run_eval()