import Preprocessing.proc_CAD.proc_gen
import Preprocessing.proc_CAD.CAD_to_stroke_cloud
import Preprocessing.proc_CAD.render_images
import Preprocessing.proc_CAD.Program_to_STL
import Preprocessing.proc_CAD.helper
import Preprocessing.proc_CAD.render_images
import Preprocessing.proc_CAD.draw_all_lines

import Preprocessing.gnn_graph
import Preprocessing.SBGCN.run_SBGCN
import Preprocessing.SBGCN.brep_read

import shutil
import os
import pickle
import torch
import numpy as np

class dataset_generator():

    def __init__(self):
        # if os.path.exists('dataset'):
        #     shutil.rmtree('dataset')
        # os.makedirs('dataset', exist_ok=True)

        self.generate_dataset('dataset/test', number_data = 5, start = 0)
        self.generate_dataset('dataset/simple', number_data = 0, start = 1245)
        self.generate_dataset('dataset/eval', number_data = 0, start = 0)


    def generate_dataset(self, dir, number_data, start):
        successful_generations = start

        while successful_generations < number_data:
            if self.generate_single_data(successful_generations, dir):
                successful_generations += 1
            else:
                print("Retrying...")


    def generate_single_data(self, successful_generations, dir):
        data_directory = os.path.join(dir, f'data_{successful_generations}')
        if os.path.exists(data_directory):
            shutil.rmtree(data_directory)

        os.makedirs(data_directory, exist_ok=True)
        
        # Generate a new program & save the brep
        try:
            # Pass in the directory to the simple_gen function
            Preprocessing.proc_CAD.proc_gen.random_program(data_directory)
            # Preprocessing.proc_CAD.proc_gen.simple_gen(data_directory)

            # Create brep for the new program and pass in the directory
            valid_parse = Preprocessing.proc_CAD.Program_to_STL.run(data_directory)
        except Exception as e:
            print(f"An error occurred: {e}")
            shutil.rmtree(data_directory)
            return False
        
        if not valid_parse:
            shutil.rmtree(data_directory)
            return False
        
        
        stroke_cloud_class = Preprocessing.proc_CAD.draw_all_lines.create_stroke_cloud_class(data_directory)

        brep_directory = os.path.join(data_directory, 'canvas')
        brep_files = [file_name for file_name in os.listdir(brep_directory) if file_name.startswith('brep_') and file_name.endswith('.step')]
        brep_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))


        prev_stop_idx = 0
        file_count = 0

        while True:
            # 1) Produce the Stroke Cloud features
            next_stop_idx = stroke_cloud_class.get_next_stop()
            if next_stop_idx == -1 or next_stop_idx > len(brep_files):
                break 
            
            
            stroke_cloud_class.read_next(next_stop_idx)
            stroke_node_features, stroke_operations_order_matrix= Preprocessing.gnn_graph.build_graph(stroke_cloud_class.edges)
            stroke_node_features = np.round(stroke_node_features, 4)
            # stroke_node_features = stroke_node_features[:, :-1]


            # 2) Get the loops
            stroke_cloud_loops = Preprocessing.proc_CAD.helper.face_aggregate_networkx(stroke_node_features)
            stroke_cloud_loops = Preprocessing.proc_CAD.helper.reorder_loops(stroke_cloud_loops)


            # 3) Compute Loop Information
            loop_neighboring_all = Preprocessing.proc_CAD.helper.loop_neighboring_simple(stroke_cloud_loops)
            loop_neighboring_vertical = Preprocessing.proc_CAD.helper.loop_neighboring_complex(stroke_cloud_loops, stroke_node_features)
            loop_neighboring_horizontal = Preprocessing.proc_CAD.helper.coplanr_neighorbing_loop(loop_neighboring_all, loop_neighboring_vertical)
            loop_neighboring_contained = Preprocessing.proc_CAD.helper.loop_contained(stroke_cloud_loops, stroke_node_features)
            
            # 4) Load Brep
            # brep_edges = stroke_cloud_class.brep_edges
            if prev_stop_idx == 0:
                final_brep_edges = np.zeros(0)
                brep_loops = []
                brep_loop_neighboring = np.zeros(0)
                stroke_to_brep = np.zeros(0)
            
            else:
                usable_brep_files = brep_files[:prev_stop_idx]
                final_brep_edges_list = []
                prev_brep_edges = []

                for file_name in usable_brep_files:
                    brep_file_path = os.path.join(brep_directory, file_name)
                    edge_features_list, edge_coplanar_list= Preprocessing.SBGCN.brep_read.create_graph_from_step_file(brep_file_path)
                    if len(prev_brep_edges) == 0:
                        final_brep_edges_list = edge_features_list
                        prev_brep_edges = edge_features_list
                        new_features = edge_features_list
                    else:
                        # We already have brep
                        new_features= find_new_features(prev_brep_edges, edge_features_list) 
                        final_brep_edges_list += new_features
                        prev_brep_edges = edge_features_list
                


                # brep_file_path = os.path.join(brep_directory, usable_brep_files[-1])
                # final_brep_edges_list, _ = Preprocessing.SBGCN.brep_read.create_graph_from_step_file(brep_file_path)

                final_brep_edges = np.array(final_brep_edges_list)
                final_brep_edges = np.round(final_brep_edges, 4)
                brep_loops = Preprocessing.proc_CAD.helper.face_aggregate_networkx(final_brep_edges)
                brep_loop_neighboring = Preprocessing.proc_CAD.helper.loop_neighboring_simple(brep_loops)


                # 5) Stroke_Cloud - Brep Connection
                stroke_to_brep = Preprocessing.proc_CAD.helper.stroke_to_brep(stroke_cloud_loops, brep_loops, stroke_node_features, final_brep_edges)


            # 6) Update the next brep file to read
            prev_stop_idx = next_stop_idx+1


            # 7) Write the data to file
            os.makedirs(os.path.join(data_directory, 'shape_info'), exist_ok=True)
            output_file_path = os.path.join(data_directory, 'shape_info', f'shape_info_{file_count}.pkl')
            with open(output_file_path, 'wb') as f:
                pickle.dump({
                    'stroke_cloud_loops': stroke_cloud_loops, 
                    'brep_loops': brep_loops,

                    'stroke_node_features': stroke_node_features,
                    'final_brep_edges': final_brep_edges,
                    'stroke_operations_order_matrix': stroke_operations_order_matrix, 

                    'loop_neighboring_vertical': loop_neighboring_vertical,
                    'loop_neighboring_horizontal': loop_neighboring_horizontal,
                    'loop_neighboring_contained': loop_neighboring_contained,

                    'brep_loop_neighboring': brep_loop_neighboring,

                    'stroke_to_brep': stroke_to_brep
                }, f)
            
            file_count += 1

        return True





def find_new_features(final_brep_edges, edge_features_list):
    def is_same_direction(line1, line2):
        """Check if two lines have the same direction."""
        vector1 = np.array(line1[3:]) - np.array(line1[:3])
        vector2 = np.array(line2[3:]) - np.array(line2[:3])
        return np.allclose(vector1 / np.linalg.norm(vector1), vector2 / np.linalg.norm(vector2))

    def is_point_on_line(point, line):
        """Check if a point lies on a given line."""
        start, end = np.array(line[:3]), np.array(line[3:])
        return np.allclose(np.cross(end - start, point - start), 0)

    def is_line_contained(line1, line2):
        """Check if line1 is contained within line2."""
        return is_point_on_line(np.array(line1[:3]), line2) and is_point_on_line(np.array(line1[3:]), line2)

    def replace_line_in_faces(faces, old_line, new_line):
        """Replace the old line with the new line in all faces."""
        for face in faces:
            for i in range(len(face)):
                if np.allclose(face[i], old_line):
                    face[i] = new_line

    new_features = []

    for edge_line in edge_features_list:
        relation_found = False

        for brep_line in final_brep_edges:
            if np.allclose(edge_line, brep_line):
                # Relation 1: The two lines are exactly the same
                relation_found = True
                break
            
            elif is_same_direction(edge_line, brep_line) and is_line_contained(brep_line, edge_line):
                # Relation 2: edge_features_list line contains final_brep_edges line
                relation_found = True
                
                if np.allclose(edge_line[:3], brep_line[:3]) or np.allclose(edge_line[:3], brep_line[3:]):
                    new_line = brep_line[3:] + edge_line[3:]
                else:
                    new_line = brep_line[:3] + edge_line[3:]

                new_features.append(new_line)
                break
            
            elif is_same_direction(edge_line, brep_line) and is_line_contained(edge_line, brep_line):
                # Relation 3: final_brep_edges line contains edge_features_list line
                relation_found = True
                
                if np.allclose(edge_line[:3], brep_line[:3]) or np.allclose(edge_line[:3], brep_line[3:]):
                    new_line = edge_line[3:] + brep_line[3:]
                else:
                    new_line = edge_line[:3] + brep_line[3:]

                new_features.append(new_line)
                break
        
        if not relation_found:
            # Relation 4: None of the relations apply
            new_features.append(edge_line)


    return new_features