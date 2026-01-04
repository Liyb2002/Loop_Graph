# CADrawer: Autoregressive Generation of CAD Program

This is the official implementation for the paper **CADrawer: Autoregressive Generation of CAD Program**.  
Our system takes **3D sketches** as input and outputs **CAD programs**.

Paper Link : [TBD]

If you have any problems, feel free to ask help from yuanboli at brown dot edu.

---

## Prerequisite

```bash
conda env create -f environment.yml
```

## Dataset

### Dataset A

Dataset A is originally created by **CAD2Sketch**.

- **Paper:** https://repo-sam.inria.fr/d3/cad2sketch/cad2sketch_paper.pdf  
- **Original dataset:** https://repo-sam.inria.fr/d3/cad2sketch/cad2sketch_dataset.g

However, the original dataset only contains CAD programs and does not include the B-rep generation process.  
It also does not contain perturbations of strokes.

We implement our own version of Dataset A, which includes the B-rep generation process and stroke perturbations.  
It can be found at: https://drive.google.com/drive/folders/1eEomAZ0Ju_r_dClBIOx5vRbvfPeKS6Gk?usp=drive_link


### Dataset B

Dataset B can be procedurally generated using our code.

To generate Dataset B, run:

```bash
python generate_dataset.py
```

For more control over the number of data samples and program length, please consult:
preprocessing/generate_dataset_baseline.py



## Testing

You may find our trained model weights at:

https://drive.google.com/drive/folders/1b8nFyN5CjrROkTvIogoKSCM5nBmydPzm?usp=sharing

Please copy the entire folder and place it under: /checkpoints

Run the following command for testing: 
```bash
python generate_CAD_program.py
```

Results will be saved to: /program_output_dataset


## Training
Please run the following script to train all required models: 
```bash
run_predictions.sh
```

## UI
For testing on extra self-designed data, please use the following system to input and uplift sketches to 3D:

- **Paper:** [https://repo-sam.inria.fr/d3/cad2sketch/cad2sketch_paper.pdf](https://www-sop.inria.fr/reves/Basilic/2025/WB25/)  
- **Github repo:** [https://repo-sam.inria.fr/d3/cad2sketch/cad2sketch_dataset.g](https://gitlab.inria.fr/D3/blender-addon-symmetry-sketch)



