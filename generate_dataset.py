
import Preprocessing.perturb_stroke_cloud
import Preprocessing.process_felix_dataset


def run():
    
    # d_generator = Preprocessing.perturb_stroke_cloud.perturbation_dataset_loader()
    d_generator = Preprocessing.process_felix_dataset.cad2sketch_dataset_loader()

    # cad2sketch_generator = Preprocessing.cad2sketch_data_generator.cad2sketch_dataset_generator()



if __name__ == "__main__":
    run()
