import json
from Preprocessing.proc_CAD.basic_class import Face, Edge, Vertex
import Preprocessing.proc_CAD.line_utils

import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.interpolate import CubicSpline


class create_stroke_cloud():
    def __init__(self, file_path, brep_edges, output = True):
        self.file_path = file_path

        self.order_count = 0
        self.faces = {}
        self.edges = {}
        self.vertices = {}
        self.id_to_count = {}

        self.brep_edges = brep_edges
        
        
    def read_json_file(self):
        with open(self.file_path, 'r') as file:
            data = json.load(file)
            for index, op in enumerate(data):
                self.parse_op(op, index)
            
        
        self.adj_edges()
        self.map_id_to_count()
        
        return


    def output(self, onlyStrokes = True):
        print("Outputting details of all components...")

        # Output vertices
        print("\nVertices:")
        if not onlyStrokes:
            for vertex_id, vertex in self.vertices.items():
                print(f"Vertex ID: {vertex_id}, Position: {vertex.position}")

            # Output faces
            print("\nFaces:")
            for face_id, face in self.faces.items():
                vertex_ids = [vertex.id for vertex in face.vertices]
                normal = face.normal
                print(f"Face ID: {face_id}, Vertices: {vertex_ids}, Normal: {normal}")


        # Output edges
        print("\nEdges:")
        for edge_id, edge in self.edges.items():
            vertex_ids = [vertex.id for vertex in edge.vertices]
            # Adding checks if 'Op' and 'order_count' are attributes of edge
            ops = getattr(edge, 'Op', 'No operations')
            order_count = getattr(edge, 'order_count', 'No order count')
            connected_edge_ids = getattr(edge, 'connected_edges', None)
        
            print(f"Edge ID: {edge_id}, Vertices: {vertex_ids},  Operations: {ops}, Order Count: {order_count}, Connected Edges: {connected_edge_ids}")


    def vis_stroke_cloud(self, directory, show=False, target_Op=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Remove grid and axes
        ax.grid(False)
        ax.set_axis_off()

        # Initialize min and max limits
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
        z_min, z_max = float('inf'), float('-inf')

        for _, edge in self.edges.items():
            # Determine line color, alpha, and thickness based on edge type
            if edge.edge_type == 'feature_line':
                line_color = 'black'
                line_alpha = edge.alpha_value
                line_thickness = np.random.uniform(0.7, 0.9)
            elif edge.edge_type == 'construction_line':
                line_color = 'black'
                line_alpha = edge.alpha_value
                line_thickness = np.random.uniform(0.4, 0.6)

            # Get edge points and perturb them to create a hand-drawn effect
            points = [vertex.position for vertex in edge.vertices]
            if len(points) == 2:
                # Original points
                x_values = np.array([points[0][0], points[1][0]])
                y_values = np.array([points[0][1], points[1][1]])
                z_values = np.array([points[0][2], points[1][2]])

                # Update min and max limits for each axis
                x_min, x_max = min(x_min, x_values.min()), max(x_max, x_values.max())
                y_min, y_max = min(y_min, y_values.min()), max(y_max, y_values.max())
                z_min, z_max = min(z_min, z_values.min()), max(z_max, z_values.max())

                # Add small random perturbations to make the line appear hand-drawn
                perturb_factor = 0.002  # Adjusted perturbation factor
                perturbations = np.random.normal(0, perturb_factor, (10, 3))  # 10 intermediate points

                # Create interpolated points for smoother curves
                t = np.linspace(0, 1, 10)  # Parameter for interpolation
                x_interpolated = np.linspace(x_values[0], x_values[1], 10) + perturbations[:, 0]
                y_interpolated = np.linspace(y_values[0], y_values[1], 10) + perturbations[:, 1]
                z_interpolated = np.linspace(z_values[0], z_values[1], 10) + perturbations[:, 2]

                # Use cubic splines to smooth the perturbed line
                cs_x = CubicSpline(t, x_interpolated)
                cs_y = CubicSpline(t, y_interpolated)
                cs_z = CubicSpline(t, z_interpolated)

                # Smooth curve points
                smooth_t = np.linspace(0, 1, 100)
                smooth_x = cs_x(smooth_t)
                smooth_y = cs_y(smooth_t)
                smooth_z = cs_z(smooth_t)

                # Plot edges with randomized line thickness and alpha
                ax.plot(smooth_x, smooth_y, smooth_z, color=line_color, alpha=line_alpha, linewidth=line_thickness)

        # Compute the center of the shape
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2

        # Compute the maximum difference across x, y, z directions
        max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)

        # Set the same limits for x, y, and z axes centered around the computed center
        ax.set_xlim([x_center - max_diff / 2, x_center + max_diff / 2])
        ax.set_ylim([y_center - max_diff / 2, y_center + max_diff / 2])
        ax.set_zlim([z_center - max_diff / 2, z_center + max_diff / 2])

        if show:
            plt.show()

        filepath = os.path.join(directory, '3d_visualization.png')
        plt.savefig(filepath)
        plt.close(fig)


    def vis_brep(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Initialize min and max limits for each axis
        x_min, x_max = np.inf, -np.inf
        y_min, y_max = np.inf, -np.inf
        z_min, z_max = np.inf, -np.inf

        # Plot all edges
        for edge in self.brep_edges:
            x_values = np.array([edge[0], edge[3]])
            y_values = np.array([edge[1], edge[4]])
            z_values = np.array([edge[2], edge[5]])

            # Plot the line in black
            ax.plot(x_values, y_values, z_values, color='black')

            # Update min and max limits for each axis
            x_min, x_max = min(x_min, x_values.min()), max(x_max, x_values.max())
            y_min, y_max = min(y_min, y_values.min()), max(y_max, y_values.max())
            z_min, z_max = min(z_min, z_values.min()), max(z_max, z_values.max())

        # Compute the center of the shape
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2

        # Compute the maximum difference across x, y, z directions
        max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)

        # Set the same limits for x, y, and z axes centered around the computed center
        ax.set_xlim([x_center - max_diff / 2, x_center + max_diff / 2])
        ax.set_ylim([y_center - max_diff / 2, y_center + max_diff / 2])
        ax.set_zlim([z_center - max_diff / 2, z_center + max_diff / 2])

        # Display the plot
        plt.show()

    
    def parse_op(self, Op, index):
        op = Op['operation'][0]

        if op == 'terminate':
            construction_lines = Preprocessing.proc_CAD.line_utils.whole_bounding_box_lines(self.edges)
            for line in construction_lines:
                line.set_edge_type('construction_line')
                line.set_order_count(self.order_count)
                line.set_Op(op, index)
                self.order_count += 1
                self.edges[line.id] = line
            
            self.determine_edge_type()
        
            for edge_id, edge in self.edges.items():
                edge.set_alpha_value()

            # self.edges = proc_CAD.line_utils.remove_duplicate_lines(self.edges)
            # self.edges = Preprocessing.proc_CAD.line_utils.perturbing_lines(self.edges)
            return

        for vertex_data in Op['vertices']:
            vertex = Vertex(id=vertex_data['id'], position=vertex_data['coordinates'])
            self.vertices[vertex.id] = vertex


        cur_op_vertex_ids = []
        new_edges = []
        for edge_data in Op['edges']:
            vertices = [self.vertices[v_id] for v_id in edge_data['vertices']]

            for v_id in edge_data['vertices']:
                cur_op_vertex_ids.append(v_id)

            edge = Edge(id=edge_data['id'], vertices=vertices)
            edge.set_Op(op, index)
            edge.set_order_count(self.order_count)
            new_edges.append(edge)
            self.order_count += 1
            # self.edges[edge.id] = edge


        # Now add the new edges to self.edges
        self.add_new_edges(new_edges)

        construction_lines = []
        # Now, we need to generate the construction lines
        if op == 'sketch':
            construction_lines = Preprocessing.proc_CAD.line_utils.midpoint_lines(new_edges)
            construction_lines += Preprocessing.proc_CAD.line_utils.diagonal_lines(new_edges)                

        if op == 'extrude':
            construction_lines = Preprocessing.proc_CAD.line_utils.projection_lines(new_edges)
            construction_lines += Preprocessing.proc_CAD.line_utils.bounding_box_lines(new_edges)
            # construction_lines = Preprocessing.proc_CAD.line_utils.grid_lines(self.edges, new_edges)

        for line in construction_lines:
            line.set_edge_type('construction_line')
            line.set_order_count(self.order_count)
            line.set_Op(op, index)
            self.order_count += 1
            self.edges[line.id] = line
        

        #find the edges that has the current operation 
        #but not created by the current operation
        self.find_unwritten_edges(cur_op_vertex_ids, op, index)

        for face_data in Op['faces']:
            vertices = [self.vertices[v_id] for v_id in face_data['vertices']]
            normal = face_data['normal']
            face = Face(id=face_data['id'], vertices=vertices, normal=normal)
            self.faces[face.id] = face          


    def adj_edges(self):

        def vert_on_line(vertex, edge):
            # Get the two vertices of the edge
            v1, v2 = edge.vertices

            # Get positions of the vertices
            p1 = v1.position
            p2 = v2.position
            p3 = vertex.position

            # Check if the vertex is one of the line endpoints
            if p3 == p1 or p3 == p2:
                return True

            # Compute vectors
            vec1 = (p2[0] - p1[0], p2[1] - p1[1])
            vec2 = (p3[0] - p1[0], p3[1] - p1[1])

            # Check if vectors are collinear by cross product
            cross_product = vec1[0] * vec2[1] - vec1[1] * vec2[0]

            # If cross product is zero, the vectors are collinear (the vertex is on the line)
            if cross_product != 0:
                return False

            # Check if p3 is between p1 and p2 using the dot product
            dot_product = (p3[0] - p1[0]) * (p2[0] - p1[0]) + (p3[1] - p1[1]) * (p2[1] - p1[1])
            if dot_product < 0:
                return False

            squared_length = (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2
            if dot_product > squared_length:
                return False

            return True

        for edge_id, edge in self.edges.items():
            connected_edge_ids = set()  

            for vertex in edge.vertices:
                for other_edge_id, other_edge in self.edges.items():
                    if other_edge_id != edge_id and vert_on_line(vertex, other_edge):
                        connected_edge_ids.add(other_edge_id)
            
            edge.connected_edges = list(connected_edge_ids)

            # print(f"Edge {edge_id} is connected to edges: {list(connected_edge_ids)}")


    def find_unwritten_edges(self, cur_op_vertex_ids, op, index):
        vertex_id_set = set(cur_op_vertex_ids)

        for edge_id, edge in self.edges.items():
            if all(vertex.id in vertex_id_set for vertex in edge.vertices):
                edge.set_Op(op, index)

    
    def map_id_to_count(self):
        for edge_id, edge in self.edges.items():
            self.id_to_count[edge_id] = edge.order_count


    def add_new_edges(self, new_edges):
        """
        Adds new edges to the existing set of edges (self.edges).
        For each new edge:
        1) Checks if it is contained within any edge in self.edges.
        2) If not contained, adds it to self.edges.
        3) If contained, splits the existing edge and replaces it with the smallest possible edges.
        """

        # Helper function to determine if one edge is contained within another
        def is_contained(edge1, edge2):
            """Check if edge2 (q1->q2) is contained within edge1 (p1->p2)."""
            p1, p2 = edge1.vertices[0].position, edge1.vertices[1].position
            q1, q2 = edge2.vertices[0].position, edge2.vertices[1].position

            # Round positions to 4 decimal places
            p1 = tuple(round(coord, 4) for coord in p1)
            p2 = tuple(round(coord, 4) for coord in p2)
            q1 = tuple(round(coord, 4) for coord in q1)
            q2 = tuple(round(coord, 4) for coord in q2)

            # Step 1: Calculate the direction vector of edge1
            direction = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
            direction_magnitude = (direction[0]**2 + direction[1]**2 + direction[2]**2) ** 0.5
            if direction_magnitude == 0:
                return False  # Degenerate edge

            # Normalize the direction vector
            unit_dir = (direction[0] / direction_magnitude, direction[1] / direction_magnitude, direction[2] / direction_magnitude)

            # Round the normalized direction vector
            unit_dir = tuple(round(coord, 4) for coord in unit_dir)

            # Check if q1 and q2 are on the line defined by edge1
            def is_point_on_line(p, p1, unit_dir):
                """Check if point p is on the line defined by point p1 and direction vector unit_dir."""
                t_values = []
                for i in range(3):
                    if unit_dir[i] != 0:  # Avoid division by zero
                        t = (p[i] - p1[i]) / unit_dir[i]
                        t_values.append(round(t, 4))  # Round t to 4 decimals

                return all(abs(t - t_values[0]) < 1e-6 for t in t_values)

            if not (is_point_on_line(q1, p1, unit_dir) and is_point_on_line(q2, p1, unit_dir)):
                return False  # q1 or q2 is not on the line defined by edge1

            # Check if q1 and q2 are between p1 and p2
            def is_between(p, p1, p2):
                """Check if point p is between points p1 and p2."""
                return all(min(round(p1[i], 4), round(p2[i], 4)) <= round(p[i], 4) <= max(round(p1[i], 4), round(p2[i], 4)) for i in range(3))

            return is_between(q1, p1, p2) and is_between(q2, p1, p2)

        # Helper function to create or reuse vertices
        def get_or_create_vertex(position, vertices_dict):
            """Returns an existing vertex if it matches the position or creates a new one."""
            position = tuple(round(coord, 4) for coord in position)  # Round position to 4 decimals
            for vertex in vertices_dict.values():
                if vertex.position == position:
                    return vertex
            vertex_id = f"vert_{len(vertices_dict)}"
            new_vertex = Vertex(id=vertex_id, position=position)
            vertices_dict[vertex_id] = new_vertex
            return new_vertex

        # Step 1: Iterate through each new edge
        for new_edge in new_edges:
            is_edge_contained = False
            edges_to_remove = []
            edges_to_add = []

            # Check if the new edge is contained within any existing edge
            for prev_edge_id, prev_edge in list(self.edges.items()):
                if is_contained(prev_edge, new_edge):
                    # The new edge is contained within the previous edge
                    is_edge_contained = True

                    # Get positions of vertices
                    A, B = prev_edge.vertices[0].position, prev_edge.vertices[1].position
                    C, D = new_edge.vertices[0].position, new_edge.vertices[1].position

                    # Round positions to 4 decimal places
                    A = tuple(round(coord, 4) for coord in A)
                    B = tuple(round(coord, 4) for coord in B)
                    C = tuple(round(coord, 4) for coord in C)
                    D = tuple(round(coord, 4) for coord in D)

                    # Step 1: Find unique points and their order along the line
                    unique_points = {tuple(A): 'A', tuple(B): 'B', tuple(C): 'C', tuple(D): 'D'}
                    unique_positions = sorted(unique_points.keys(), key=lambda p: (p[0], p[1], p[2]))

                    # Step 2: Create or reuse vertices
                    vertex_map = {p: get_or_create_vertex(p, self.vertices) for p in unique_positions}

                    # Step 3: Create new edges for each consecutive pair of unique points
                    for i in range(len(unique_positions) - 1):
                        start = vertex_map[unique_positions[i]]
                        end = vertex_map[unique_positions[i + 1]]
                        edge_id = f"edge_{len(self.edges)}_{i}"
                        new_edge = Edge(id=edge_id, vertices=(start, end))
                        edges_to_add.append(new_edge)

                    edges_to_remove.append(prev_edge_id)
                    break  # No need to check other previous edges since it is already contained

            # Step 2: Add the new edge if not contained within any existing edge
            if not is_edge_contained:
                self.edges[new_edge.id] = new_edge
            else:
                # Remove the contained edge and add the new split edges
                for edge_id in edges_to_remove:
                    del self.edges[edge_id]
                for edge in edges_to_add:
                    self.edges[edge.id] = edge


    def determine_edge_type(self):
        """
        Determines the type of each edge in self.edges.
        For each edge with type 'maybe_feature_line':
        1) Checks if it is contained within any brep_edge in self.brep_edges.
        2) If contained, sets its type to 'feature_line'.
        3) If not contained, sets its type to 'construction_line'.
        """
        # Helper function to round a 3D point to 4 decimals
        def round_point(point):
            return tuple(round(coord, 4) for coord in point)

        # Helper function to check if an edge is contained within a brep edge
        def is_contained_in_brep(edge, brep_edge):
            """Check if edge (with two vertices) is contained within brep_edge (a list of 6 values)."""
            p1, p2 = edge.vertices[0].position, edge.vertices[1].position
            q1, q2 = tuple(brep_edge[:3]), tuple(brep_edge[3:])

            # Round the points for comparison
            p1, p2 = round_point(p1), round_point(p2)
            q1, q2 = round_point(q1), round_point(q2)

            # Check if both vertices of edge are on the brep edge
            def is_between(p, a, b):
                """Check if point p is between points a and b."""
                return all(min(a[i], b[i]) <= p[i] <= max(a[i], b[i]) for i in range(3))

            # Ensure the condition that if p1 and p2 have the same value on any axis, then q1 and q2 must also
            for i in range(3):  # Loop over x, y, z axes
                if p1[i] == p2[i]:  # Check if p1 and p2 have the same value on this axis
                    if q1[i] != q2[i]:  # If q1 and q2 do not have the same value on this axis
                        return False  # The brep edge does not satisfy the condition

            # Check if edge is contained by brep_edge or has the same vertices
            if (p1 == q1 and p2 == q2) or (p1 == q2 and p2 == q1):
                return True  # Same vertices
            elif is_between(p1, q1, q2) and is_between(p2, q1, q2):
                return True  # Both points are on the brep edge
            else:
                return False

        # Step 1: Iterate through each edge in self.edges
        for edge in self.edges.values():
            # Only process edges with type 'maybe_feature_line'
            if edge.edge_type == 'maybe_feature_line':
                contained_in_brep = False

                # Step 2: Check if this edge is contained in any brep_edge
                for brep_edge in self.brep_edges:
                    if is_contained_in_brep(edge, brep_edge):
                        edge_start = edge.vertices[0].position
                        edge_end = edge.vertices[1].position
                        
                        # Extract coordinates from the brep_edge
                        brep_start = brep_edge[:3]
                        brep_end = brep_edge[3:]

                        contained_in_brep = True
                        break

                # Step 3: Set edge type based on containment
                if contained_in_brep:
                    edge.set_edge_type('feature_line')
                else:
                    edge.set_edge_type('construction_line')



def run(directory):
    file_path = os.path.join(directory, 'Program.json')
    
    # Get the final brep file
    brep_directory = os.path.join(directory, 'canvas')
    brep_files = [file_name for file_name in os.listdir(brep_directory)
            if file_name.startswith('brep_') and file_name.endswith('.step')]
    brep_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    brep_file_name = brep_files[-1]

    # Read the brep edges
    brep_file_path = os.path.join(brep_directory, brep_file_name)
    brep_edges, _ = Preprocessing.SBGCN.brep_read.create_graph_from_step_file(brep_file_path)



    stroke_cloud_class = create_stroke_cloud(file_path, brep_edges)
    stroke_cloud_class.read_json_file()

    # stroke_cloud_class.vis_brep()
    stroke_cloud_class.vis_stroke_cloud(directory, show = True)

    return stroke_cloud_class.edges, stroke_cloud_class.faces