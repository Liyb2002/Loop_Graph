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

This code is build on certain modules from CAD2Sketch, the following command lines would be necessary for data generation and trainning, but not for quick testing:

```bash
git clone https://gitlab.inria.fr/D3/pylowstroke.git
cd pylowstroke
pip install -r requirements.txt
pip install .
cd ..

git clone https://gitlab.inria.fr/D3/blender-addon-symmetry-sketch.git
cd blender-addon-symmetry-sketch
pip install -r requirements.txt
pip install .
```

#### Quick notes: 
The code was written using Build123d==0.5.0 version. But pip might automatically install Build123d==0.9.0
The authors has adapted the code for Build123d==0.9.0, so it should work fine. But if any problem happens, please switch back to Build123d==0.5.0.
Or you may manually edit the code using Build123d in /proc_CAD/build123/protocol.py

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
This bash script will train all the required tasks networks. 

To train and evaluate a specific task's network, please refer to its trainning code (e.g fillet_prediction.py).
You may toggle the train() and eval() function at the end of the code for evaluation purposes.

To select the target dataset for trainning or evaluation purposes, please find line below to change the dataset directory:
```bash
dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/datasetB')
```

## UI
For testing on extra self-designed data, please use the following system to input and uplift sketches to 3D:

- **Paper:** [https://repo-sam.inria.fr/d3/cad2sketch/cad2sketch_paper.pdf](https://www-sop.inria.fr/reves/Basilic/2025/WB25/)  
- **Github repo:** [https://repo-sam.inria.fr/d3/cad2sketch/cad2sketch_dataset.g](https://gitlab.inria.fr/D3/blender-addon-symmetry-sketch)



