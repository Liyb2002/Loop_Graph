
import Preprocessing.perturb_stroke_cloud
import Preprocessing.process_felix_dataset
import Preprocessing.perturb_synthetic
import Preprocessing.generate_dataset_baseline

target = 'datasetB'
def run():
    
    # d_generator = Preprocessing.perturb_synthetic.perturbation_dataset_loader()
    # d_generator = Preprocessing.process_felix_dataset.cad2sketch_dataset_loader(target)

    # d_generator = Preprocessing.perturb_stroke_cloud.perturbation_dataset_loader(target)
    # d_generator = Preprocessing.process_felix_dataset.cad2sketch_dataset_loader(target)

    cad2sketch_generator = Preprocessing.generate_dataset_baseline.dataset_generator()



if __name__ == "__main__":
    run()
