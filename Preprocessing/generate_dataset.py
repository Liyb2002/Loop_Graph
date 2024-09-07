import Preprocessing.proc_CAD.proc_gen
import Preprocessing.proc_CAD.CAD_to_stroke_cloud
import Preprocessing.proc_CAD.render_images
import Preprocessing.proc_CAD.Program_to_STL
import Preprocessing.proc_CAD.helper
import Preprocessing.proc_CAD.render_images

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

        self.generate_dataset('dataset/test', number_data = 1, start = 0)
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
        
        
        Unfinished_shape = True
        stroke_cloud_class = Preprocessing.proc_CAD.CAD_to_stroke_cloud.create_stroke_cloud_class(data_directory)

        while Unfinished_shape:
            Unfinished_shape = stroke_cloud_class.read_next()
            stroke_cloud_edges = stroke_cloud_class.edges
            stroke_node_features, stroke_operations_order_matrix= Preprocessing.gnn_graph.build_graph(stroke_cloud_edges)









# --------------------------------------------------------------------------------------------------------


        # 1) Save matrices for stroke_cloud_graph
        stroke_cloud_edges, stroke_cloud_faces= Preprocessing.proc_CAD.CAD_to_stroke_cloud.run(data_directory)
        node_features, operations_order_matrix= Preprocessing.gnn_graph.build_graph(stroke_cloud_edges)
        stroke_cloud_save_path = os.path.join(data_directory, 'stroke_cloud_graph.pkl')
        
        
        face_to_stroke = Preprocessing.proc_CAD.helper.face_to_stroke(node_features)
        gnn_strokeCloud_edges = Preprocessing.proc_CAD.helper.gnn_edges(face_to_stroke)
        stroke_cloud_coplanar = Preprocessing.proc_CAD.helper.coplanar_matrix(face_to_stroke, node_features)

        with open(stroke_cloud_save_path, 'wb') as f:
            pickle.dump({
                'node_features': node_features,
                'operations_order_matrix': operations_order_matrix,
                'gnn_strokeCloud_edges': gnn_strokeCloud_edges,
                'face_to_stroke': face_to_stroke,
                'stroke_cloud_coplanar': stroke_cloud_coplanar
            }, f)


        # 3) Save matrices for Brep Embedding
        brep_directory = os.path.join(data_directory, 'canvas')
        brep_files = [file_name for file_name in os.listdir(brep_directory)
              if file_name.startswith('brep_') and file_name.endswith('.step')]
        brep_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))


        final_brep_edges = []
        prev_brep_edges = []
        for file_name in brep_files:
            brep_file_path = os.path.join(brep_directory, file_name)
            edge_features_list, edge_coplanar_list= Preprocessing.SBGCN.brep_read.create_graph_from_step_file(brep_file_path)

            # If this is the first brep
            if len(prev_brep_edges) == 0:
                final_brep_edges = edge_features_list
                prev_brep_edges = edge_features_list
                new_features = edge_features_list
            else:
                # We already have brep
                new_features= find_new_features(prev_brep_edges, edge_features_list) 
                final_brep_edges += new_features
                prev_brep_edges = edge_features_list
            
            brep_to_stroke = Preprocessing.proc_CAD.helper.face_to_stroke(final_brep_edges)
            gnn_brep_edges = Preprocessing.proc_CAD.helper.gnn_edges(brep_to_stroke)

            print("face_to_stroke", face_to_stroke)
            print("brep_to_stroke", brep_to_stroke)
            print("node_features", node_features.shape)
            print("final_brep_edges", final_brep_edges)
            brep_stroke_connection = Preprocessing.proc_CAD.helper.stroke_to_brep(face_to_stroke, brep_to_stroke, node_features, final_brep_edges)
            brep_coplanar = Preprocessing.proc_CAD.helper.coplanar_matrix(brep_to_stroke, final_brep_edges)
        
       
            # extract index i
            index = file_name.split('_')[1].split('.')[0]
            os.makedirs(os.path.join(data_directory, 'brep_embedding'), exist_ok=True)
            embeddings_file_path = os.path.join(data_directory, 'brep_embedding', f'brep_info_{index}.pkl')
            with open(embeddings_file_path, 'wb') as f:
                pickle.dump({
                    'brep_to_stroke': brep_to_stroke, 
                    'edge_features': final_brep_edges,
                    'gnn_brep_edges': gnn_brep_edges,
                    'brep_stroke_connection': brep_stroke_connection,
                    'brep_coplanar': brep_coplanar

                }, f)

        # 4) Save rendered 2D image
        Preprocessing.proc_CAD.render_images.run_render_images(data_directory)


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
