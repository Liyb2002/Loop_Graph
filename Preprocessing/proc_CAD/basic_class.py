
class Face:
    def __init__(self, id, vertices, normal):
        # print(f"An Face is created with ID: {id}")
        self.id = id
        self.vertices = vertices
        self.normal = normal

        self.future_sketch = True
        self.is_cirlce = False
    
    def face_fixed(self):
        self.future_sketch = False

    def circle(self, radius, center):
        self.is_cirlce = True
        self.radius = radius
        self.center = center



class Edge:
    def __init__(self, id, vertices):
        # print(f"An edge is created with ID: {id}")
        self.id = id
        self.vertices = vertices
        self.round = False

        self.Op = []
        self.Op_orders = []
        self.order_count = 0
        self.connected_edges = []
    
    def fillet_edge(self):
        self.round = True
    
    def set_Op(self, Op, order_count):
        self.Op.append(Op)
        self.Op_orders.append(order_count)
    
    def set_order_count(self, order_count):
        self.order_count = order_count

    def connected_edges(self, edge_id):
        self.connected_edges.append(edge_id)


class Vertex:
    def __init__(self, id, position):
        # print(f"A vertex is created with ID: {id}")
        self.id = id
        self.position = position


