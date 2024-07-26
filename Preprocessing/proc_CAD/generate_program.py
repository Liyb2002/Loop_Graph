

import json
import numpy as np
import Preprocessing.proc_CAD.helper
import random
import Preprocessing.proc_CAD.random_gen

import os
from Preprocessing.proc_CAD.basic_class import Face, Edge, Vertex

class Brep:
    def __init__(self):
        self.Faces = []
        self.Edges = []
        self.Vertices = []

        self.op = []
        self.idx = 0
        
    
    def init_sketch_op(self):

        axis = np.random.choice(['x', 'y', 'z'])
        points, normal = Preprocessing.proc_CAD.random_gen.generate_random_rectangle(axis)

        if axis == 'x':
            boundary_points = ([1, 0, 0])
        elif axis == 'y':
            boundary_points = ([0, 1, 0])
        elif axis == 'z':
            boundary_points = ([0, 0, 1])

        self._sketch_op(points, normal, boundary_points)


    def _sketch_op(self, points, normal, boundary_points):
        vertex_list = []
        for i, point in enumerate(points):
            vertex_id = f"vertex_{self.idx}_{i}"
            vertex = Vertex(vertex_id, point.tolist())
            self.Vertices.append(vertex)
            vertex_list.append(vertex)

        num_vertices = len(vertex_list)
        for i in range(num_vertices):
            edge_id = f"edge_{self.idx}_{i}"
            edge = Edge(edge_id, [vertex_list[i], vertex_list[(i+1) % num_vertices]])  # Loop back to first vertex to close the shape
            edge.fillet_edge()
            self.Edges.append(edge)

        face_id = f"face_{self.idx}_{0}"
        face = Face(face_id, vertex_list, normal)
        self.Faces.append(face)
        
        self.idx += 1
        self.op.append(['sketch', boundary_points])


    def regular_sketch_op(self):

        faces_with_future_sketch = [face for face in self.Faces if face.future_sketch ]
        if not faces_with_future_sketch:
            return False
        target_face = random.choice(faces_with_future_sketch)
        target_face.face_fixed()

        boundary_points = [vert.position for vert in target_face.vertices]
        normal = [ 0 - normal for normal in target_face.normal]

        # cases = ['create_circle', 'find_rectangle', 'find_triangle', 'triangle_to_cut']
        cases = ['find_rectangle', 'find_triangle', 'triangle_to_cut']
        selected_case = random.choice(cases)
        if selected_case == 'create_circle':
            radius = Preprocessing.proc_CAD.random_gen.generate_random_cylinder_radius()
            center = Preprocessing.proc_CAD.helper.random_circle(boundary_points, normal)
            self.regular_sketch_circle(normal, radius, center)
            return 

        if selected_case == 'find_rectangle':
            random_polygon_points = Preprocessing.proc_CAD.helper.find_rectangle_on_plane(boundary_points, normal)

        if selected_case == 'find_triangle':
            random_polygon_points = Preprocessing.proc_CAD.helper.find_triangle_on_plane(boundary_points, normal)

        if selected_case == 'triangle_to_cut':
            random_polygon_points = Preprocessing.proc_CAD.helper.find_triangle_to_cut(boundary_points, normal)

        self._sketch_op(random_polygon_points, normal, boundary_points)


    def regular_sketch_circle(self, normal, radius, center):
        face_id = f"face_{self.idx}_{0}"
        face = Face(face_id, [], normal)
        face.circle(radius, center)
        self.Faces.append(face)
        
        self.idx += 1
        self.op.append(['sketch'])


    def extrude_op(self, amount = 0, sketch_face_opposite_normal = [0,0,0]):
        
        if amount == 0:
            amount = Preprocessing.proc_CAD.random_gen.generate_random_extrude()
        if self.idx < 2:
            amount = abs(amount)
        
        sketch_face = self.Faces[-1]
            
        if sketch_face_opposite_normal == [0,0,0]:
            sketch_face_opposite_normal = [-x for x in sketch_face.normal]
        else:
            sketch_face.normal = [-x for x in sketch_face_opposite_normal]

        new_vertices = []
        new_edges = []
        new_faces = []

        for i, vertex in enumerate(sketch_face.vertices):

            new_pos = [vertex.position[j] + sketch_face_opposite_normal[j] * amount for j in range(3)]
            vertex_id = f"vertex_{self.idx}_{i}"
            new_vertex = Vertex(vertex_id, new_pos)
            self.Vertices.append(new_vertex)
            new_vertices.append(new_vertex)

        num_vertices = len(new_vertices)
        for i in range(num_vertices):
            edge_id = f"edge_{self.idx}_{i}"
            edge = Edge(edge_id, [new_vertices[i], new_vertices[(i+1) % num_vertices]])  # Loop back to first vertex to close the shape
            self.Edges.append(edge)
            new_edges.append(edge)

        face_id = f"face_{self.idx}_{0}"
        new_face = Face(face_id, new_vertices, sketch_face_opposite_normal)
        self.Faces.append(new_face)
        new_faces.append(new_face)
        
        
        #create side edges and faces
        for i in range(num_vertices):
            # Vertical edges from old vertices to new vertices
            vertical_edge_id = f"edge_{self.idx}_{i+num_vertices}"
            vertical_edge = Edge(vertical_edge_id, [sketch_face.vertices[i], new_vertices[i]])
            self.Edges.append(vertical_edge)

            # Side faces formed between pairs of old and new vertices
            side_face_id = f"face_{self.idx}_{i}"
            side_face_vertices = [
                sketch_face.vertices[i], new_vertices[i],
                new_vertices[(i + 1) % num_vertices], sketch_face.vertices[(i + 1) % num_vertices]
            ]
            normal = Preprocessing.proc_CAD.helper.compute_normal(side_face_vertices, new_vertices[(i + 2) % num_vertices])
            side_face = Face(side_face_id, side_face_vertices, normal)
            self.Faces.append(side_face)

        self.idx += 1
        self.op.append(['extrude', sketch_face.id, amount])


    def random_fillet(self):
        
        edge_with_round = [edge for edge in self.Edges if not edge.round]
        if not edge_with_round:
            return False
        target_edge = random.choice(edge_with_round)

        amount = Preprocessing.proc_CAD.random_gen.generate_random_fillet()
        target_edge.fillet_edge()

        verts_pos = []
        verts_id = []
        new_vert_pos = []

        for vert in target_edge.vertices:
            verts_pos.append(vert.position)
            verts_id.append(vert.id)
            neighbor_verts = Preprocessing.proc_CAD.helper.get_neighbor_verts(vert,target_edge,  self.Edges)
            new_vert_pos.append(Preprocessing.proc_CAD.helper.compute_fillet_new_vert(vert, neighbor_verts, amount))
        
        new_A = new_vert_pos[0][0]
        new_B = new_vert_pos[0][1]
        new_C = new_vert_pos[1][0]
        new_D = new_vert_pos[1][1]


        #create 4 new verts from new_A, new_B and new_C, new_D
        new_vert_B = Vertex(f"vertex_{self.idx}_0", new_B)
        new_vert_D = Vertex(f"vertex_{self.idx}_1", new_D)
        new_vert_A = Vertex(f"vertex_{self.idx}_2", new_A)
        new_vert_C = Vertex(f"vertex_{self.idx}_3", new_C)
        self.Vertices.append(new_vert_B)
        self.Vertices.append(new_vert_D)
        self.Vertices.append(new_vert_A)
        self.Vertices.append(new_vert_C)


        #create 2 edge that connect new_B and new_D / new_A and new_C
        new_edge_id_0 = f"edge_{self.idx}_0"
        new_edge_0 = Edge(new_edge_id_0, [new_vert_B, new_vert_D])
        new_edge_id_1 = f"edge_{self.idx}_1"
        new_edge_1 = Edge(new_edge_id_1, [new_vert_A, new_vert_C])
        self.Edges.append(new_edge_0)
        self.Edges.append(new_edge_1)

        #create 2 edge that connect new_A and new_B / new_C and new_D
        new_edge_id_2 = f"edge_{self.idx}_2"
        new_edge_2 = Edge(new_edge_id_2, [new_vert_A, new_vert_B])
        new_edge_id_3 = f"edge_{self.idx}_3"
        new_edge_3 = Edge(new_edge_id_3, [new_vert_C, new_vert_D])
        self.Edges.append(new_edge_2)
        self.Edges.append(new_edge_3)

        self.idx += 1
        self.op.append(['fillet', 
                        target_edge.id, 
                        {'amount': amount}, 
                        {'old_verts_pos': verts_pos},
                        {'verts_id': verts_id},
                        ])


    def write_to_json(self, data_directory = None):
        
        #clean everything in the folder
        folder = os.path.join(os.path.dirname(__file__), 'canvas')
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        if data_directory:
            folder = data_directory

        
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

        #start writing program
        filename = os.path.join(folder, 'Program.json')
        data = []
        for count in range(0, self.idx):
            op = self.op[count][0]
            self.write_Op(self.op[count], count, data)

        self.write_terminate(data)        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"Data saved to {filename}")


    def write_Op(self, Op, index, data):
        operation = {
            'operation': Op,
            'faces': [],
            'edges': [],
            'vertices': []
        }

                # Add each point with an ID to the vertices list
        
        for face in self.Faces:
            if face.id.split('_')[1] == str(index):

                if face.is_cirlce:
                    face = {
                    'id': face.id,
                    'radius': face.radius,
                    'center': [pt for pt in face.center],
                    'normal': [float(n) if isinstance(n, np.floating) else int(n) for n in face.normal]
                    }
                else:
                    face = {
                        'id': face.id,
                        'vertices': [vertex.id for vertex in face.vertices],
                        'normal': [float(n) if isinstance(n, np.floating) else int(n) for n in face.normal]
                    }

                operation['faces'].append(face)

        for edge in self.Edges:
            if edge.id.split('_')[1] == str(index):
                
                edge = {
                    'id': edge.id,
                    'vertices': [vertex.id for vertex in edge.vertices]
                }
                operation['edges'].append(edge)


        
        for vertex in self.Vertices:
            if vertex.id.split('_')[1] == str(index):
                vertex = {
                    'id': vertex.id,
                    'coordinates': vertex.position  # Convert numpy array to list for JSON serialization
                }
                operation['vertices'].append(vertex)
        
        data.append(operation)

        return data
                

    def write_terminate(self, data):
        operation = {
            'operation': ['terminate']
        }
        data.append(operation)
        return data
