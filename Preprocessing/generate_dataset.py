import Preprocessing.proc_CAD.proc_gen
import Preprocessing.proc_CAD.CAD_to_stroke_cloud
import Preprocessing.proc_CAD.render_images
import Preprocessing.proc_CAD.Program_to_STL
import Preprocessing.proc_CAD.helper
import Preprocessing.proc_CAD.render_images
import Preprocessing.proc_CAD.draw_all_lines

import Preprocessing.gnn_graph
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

        self.generate_dataset('dataset/test', number_data = 10, start = 0)
        # self.generate_dataset('dataset/messy_order_eval', number_data = 100, start = 0)
        # self.generate_dataset('dataset/messy_order_full', number_data = -1, start = 0)


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
            print("not valid valid_parse")
            shutil.rmtree(data_directory)
            return False
        
        
        print("----------------------")
        stroke_cloud_class = Preprocessing.proc_CAD.draw_all_lines.create_stroke_cloud_class(data_directory, False)

        brep_directory = os.path.join(data_directory, 'canvas')
        brep_files = [file_name for file_name in os.listdir(brep_directory) if file_name.startswith('brep_') and file_name.endswith('.step')]
        brep_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))


        prev_stop_idx = 0
        file_count = 0

        while True:
            # 1) Produce the Stroke Cloud features
            next_stop_idx = stroke_cloud_class.get_next_stop()
            if next_stop_idx == -1 or next_stop_idx >= len(brep_files):
                break 
            
            
            stroke_cloud_class.read_next(next_stop_idx)
            stroke_node_features, stroke_operations_order_matrix= Preprocessing.gnn_graph.build_graph(stroke_cloud_class.edges)
            stroke_node_features, stroke_operations_order_matrix = Preprocessing.proc_CAD.helper.swap_rows_with_probability(stroke_node_features, stroke_operations_order_matrix)
            stroke_node_features = np.round(stroke_node_features, 4)

            connected_stroke_nodes = Preprocessing.proc_CAD.helper.connected_strokes(stroke_node_features)
            strokes_perpendicular, strokes_non_perpendicular =  Preprocessing.proc_CAD.helper.stroke_relations(stroke_node_features, connected_stroke_nodes)

            # stroke_node_features = stroke_node_features[:, :-1]


            # 2) Get the loops
            stroke_cloud_loops = Preprocessing.proc_CAD.helper.face_aggregate_networkx(stroke_node_features)
            stroke_cloud_loops = Preprocessing.proc_CAD.helper.reorder_loops(stroke_cloud_loops)


            # 3) Compute Loop Information
            loop_neighboring_all = Preprocessing.proc_CAD.helper.loop_neighboring_simple(stroke_cloud_loops)
            loop_neighboring_vertical = Preprocessing.proc_CAD.helper.loop_neighboring_complex(stroke_cloud_loops, stroke_node_features)
            loop_neighboring_horizontal = Preprocessing.proc_CAD.helper.coplanr_neighorbing_loop(loop_neighboring_all, loop_neighboring_vertical)
            loop_neighboring_contained = Preprocessing.proc_CAD.helper.loop_contained(stroke_cloud_loops, stroke_node_features)
            loop_neighboring_coplanar = Preprocessing.proc_CAD.helper.loop_coplanar(stroke_cloud_loops, stroke_node_features)

            # 4) Load Brep
            # brep_edges = stroke_cloud_class.brep_edges
            if prev_stop_idx == 0:
                final_brep_edges = np.zeros(0)
                brep_loops = []
                brep_loop_neighboring = np.zeros(0)
                stroke_to_loop = np.zeros(0)
                stroke_to_edge = np.zeros(0)
            
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
                stroke_to_loop = Preprocessing.proc_CAD.helper.stroke_to_brep(stroke_cloud_loops, brep_loops, stroke_node_features, final_brep_edges)
                stroke_to_edge = Preprocessing.proc_CAD.helper.stroke_to_edge(stroke_node_features, final_brep_edges)

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
                    'strokes_perpendicular': strokes_perpendicular,
                    'final_brep_edges': final_brep_edges,
                    'stroke_operations_order_matrix': stroke_operations_order_matrix, 

                    'loop_neighboring_vertical': loop_neighboring_vertical,
                    'loop_neighboring_horizontal': loop_neighboring_horizontal,
                    'loop_neighboring_contained': loop_neighboring_contained,
                    'loop_neighboring_coplanar':loop_neighboring_coplanar,

                    'brep_loop_neighboring': brep_loop_neighboring,

                    'stroke_to_loop': stroke_to_loop,
                    'stroke_to_edge': stroke_to_edge
                }, f)
            
            file_count += 1

        return True





def find_new_features(prev_brep_edges, new_edge_features):
    prev_brep_edges = [[round(coord, 4) for coord in line] for line in prev_brep_edges]
    new_edge_features = [[round(coord, 4) for coord in line] for line in new_edge_features]

    def is_same_direction(line1, line2):
        """Check if two lines have the same direction."""
        vector1 = np.array(line1[3:]) - np.array(line1[:3])
        vector2 = np.array(line2[3:]) - np.array(line2[:3])
        return np.allclose(vector1 / np.linalg.norm(vector1), vector2 / np.linalg.norm(vector2))

    def is_point_on_line(point, line):
        """Check if a point lies on a given line segment."""
        start, end = np.array(line[:3]), np.array(line[3:])
        
        # Check if the point is collinear (still important to check)
        if not np.allclose(np.cross(end - start, point - start), 0):
            return False
        
        # Check if point lies within the bounds of the line segment
        min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
        min_y, max_y = min(start[1], end[1]), max(start[1], end[1])
        min_z, max_z = min(start[2], end[2]), max(start[2], end[2])
        
        return (min_x <= point[0] <= max_x) and (min_y <= point[1] <= max_y) and (min_z <= point[2] <= max_z)

    def is_line_contained(line1, line2):
        """Check if line1 is contained within line2."""
        return is_point_on_line(np.array(line1[:3]), line2) and is_point_on_line(np.array(line1[3:]), line2)



    def find_unique_points(new_edge_line, prev_brep_line):
        """Find the two unique points between new_edge_line and prev_brep_line."""
        points = [
            tuple(new_edge_line[:3]),   # new_edge_line start
            tuple(new_edge_line[3:]),   # new_edge_line end
            tuple(prev_brep_line[:3]),  # prev_brep_line start
            tuple(prev_brep_line[3:]),  # prev_brep_line end
        ]

        # Find unique points
        unique_points = [point for point in points if points.count(point) == 1]

        # Ensure there are exactly two unique points
        if len(unique_points) == 2:
            return unique_points
        return None

    new_features = []

    for new_edge_line in new_edge_features:
        relation_found = False

        edge_start, edge_end = np.array(new_edge_line[:3]), np.array(new_edge_line[3:])

        for prev_brep_line in prev_brep_edges:

            brep_start, brep_end = np.array(prev_brep_line[:3]), np.array(prev_brep_line[3:])

            # Check if the lines are the same, either directly or in reverse order
            if (np.allclose(edge_start, brep_start) and np.allclose(edge_end, brep_end)) or \
            (np.allclose(edge_start, brep_end) and np.allclose(edge_end, brep_start)):
                # Relation 1: The two lines are exactly the same
                relation_found = True

                break
            
            elif is_same_direction(new_edge_line, prev_brep_line) and is_line_contained(new_edge_line, prev_brep_line):
                # new feature is in prev brep
                relation_found = True
                
                unique_points = find_unique_points(new_edge_line, prev_brep_line)
                if unique_points:
                    # Create a new line using the unique points
                    new_line = list(unique_points[0]) + list(unique_points[1])
                    new_features.append(new_line)
                    relation_found = True
                    break
                break
            
            elif is_same_direction(new_edge_line, prev_brep_line) and is_line_contained(prev_brep_line, new_edge_line):
                # prev brep is in new feature
                relation_found = True
                
                unique_points = find_unique_points(new_edge_line, prev_brep_line)
                if unique_points:
                    # Create a new line using the unique points
                    new_line = list(unique_points[0]) + list(unique_points[1])
                    new_features.append(new_line)
                    relation_found = True
                    break

                break
        
        if not relation_found:
            # Relation 4: None of the relations apply
            new_features.append(new_edge_line)


    return new_features