import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_kth_operation(op_to_index_matrix, k):    
    squeezed_matrix = op_to_index_matrix.squeeze(0)
    kth_operation = squeezed_matrix[:, k].unsqueeze(1)

    return kth_operation

def stroke_to_face(kth_operation, face_to_stroke):
    num_faces = len(face_to_stroke)
    face_chosen = torch.zeros((num_faces, 1), dtype=torch.float32)
    
    for i, strokes in enumerate(face_to_stroke):
        if all(kth_operation[stroke].item() == 1 for stroke in strokes):
            face_chosen[i] = 1

    return face_chosen


#------------------------------------------------------------------------------------------------------#

def face_is_not_in_brep(matrix, face_to_stroke, node_features, edge_features):
    # Find the max index in the matrix
    max_index = torch.argmax(matrix).item()
    
    # Get the strokes associated with the chosen face
    chosen_face_strokes = face_to_stroke[max_index]
    
    # Check if any of the strokes are in edge_features
    for stroke_index in chosen_face_strokes:
        stroke_value = node_features[stroke_index]
        if any(torch.equal(stroke_value, edge) for edge in edge_features):
            return False
    
    return True


def predict_face_coplanar_with_brep(matrix, face_to_stroke, brep_to_stroke, node_features, edge_features):
    # Find the max index in the matrix
    max_index = torch.argmax(matrix).item()
    
    # Get the strokes associated with the chosen face
    chosen_face_strokes = face_to_stroke[max_index]
    
    # Gather all x, y, and z values from the chosen face strokes
    x_values, y_values, z_values = set(), set(), set()
    for stroke_index in chosen_face_strokes:
        stroke = node_features[stroke_index][0]
        x_values.update([stroke[0].item(), stroke[3].item()])
        y_values.update([stroke[1].item(), stroke[4].item()])
        z_values.update([stroke[2].item(), stroke[5].item()])
    
    # Determine the common axis and value
    if len(x_values) == 1:
        axis, value = 'x', next(iter(x_values))
    elif len(y_values) == 1:
        axis, value = 'y', next(iter(y_values))
    elif len(z_values) == 1:
        axis, value = 'z', next(iter(z_values))
    else:
        return False
    
    # Check if any of the brep faces are coplanar with the chosen face
    for brep_face in brep_to_stroke:
        brep_x_values, brep_y_values, brep_z_values = set(), set(), set()
        for stroke_index in brep_face:
            stroke = edge_features[stroke_index][0]
            brep_x_values.update([stroke[0].item(), stroke[3].item()])
            brep_y_values.update([stroke[1].item(), stroke[4].item()])
            brep_z_values.update([stroke[2].item(), stroke[5].item()])
        
        if axis == 'x' and len(brep_x_values) == 1 and next(iter(brep_x_values)) == value:
            return True
        elif axis == 'y' and len(brep_y_values) == 1 and next(iter(brep_y_values)) == value:
            return True
        elif axis == 'z' and len(brep_z_values) == 1 and next(iter(brep_z_values)) == value:
            return True
    
    return False

#------------------------------------------------------------------------------------------------------#

def vis_stroke_cloud(node_features):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    node_features = node_features.squeeze(0)

    # Plot all strokes in blue
    for stroke in node_features:
        start = stroke[:3].numpy()
        end = stroke[3:].numpy()
        
        # Plot the line segment for the stroke in blue
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], marker='o', color='blue')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

    
def vis_gt(matrix, face_to_stroke, node_features):
    # Initialize a list to keep track of stroke colors
    stroke_colors = ['blue'] * node_features.shape[0]

    # Find the index of the item with the highest value in the matrix
    max_index = torch.argmax(matrix).item()

    # Set the strokes in the face with the highest value to red
    for stroke in face_to_stroke[max_index]:
        stroke_colors[stroke] = 'red'
    
    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each stroke
    for i, stroke in enumerate(node_features):
        x = [stroke[0], stroke[3]]
        y = [stroke[1], stroke[4]]
        z = [stroke[2], stroke[5]]
        ax.plot(x, y, z, color=stroke_colors[i])
    
    plt.show()


def vis_brep_and_nextSketch(matrix, face_to_stroke, node_features, edge_features):
    # Initialize a list to keep track of stroke colors
    stroke_colors = ['blue'] * node_features.shape[0]

    # Find the index of the item with the highest value in the matrix
    max_index = torch.argmax(matrix).item()

    # Set the strokes in the face with the highest value to red
    for stroke in face_to_stroke[max_index]:
        stroke_colors[stroke] = 'red'
    
    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot node_features in blue (or red if in max_index face)
    for i, stroke in enumerate(node_features):
        if stroke_colors[i] == 'blue':
            x = [stroke[0], stroke[3]]
            y = [stroke[1], stroke[4]]
            z = [stroke[2], stroke[5]]
            ax.plot(x, y, z, color=stroke_colors[i])
    
    # Plot edge_features in green
    for stroke in edge_features:
        x = [stroke[0], stroke[3]]
        y = [stroke[1], stroke[4]]
        z = [stroke[2], stroke[5]]
        ax.plot(x, y, z, color='green')

    for i, stroke in enumerate(node_features):
        if stroke_colors[i] == 'red':
            x = [stroke[0], stroke[3]]
            y = [stroke[1], stroke[4]]
            z = [stroke[2], stroke[5]]
            ax.plot(x, y, z, color=stroke_colors[i])

    plt.show()


def vis_strokes(node_features, color_matrix):
    print("color_matrix", color_matrix)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    node_features = node_features.squeeze(0)

    # Plot strokes in the specified colors
    for i, stroke in enumerate(node_features):
        start = stroke[:3].numpy()
        end = stroke[3:].numpy()
        
        # Determine color based on color_matrix
        color = 'red' if color_matrix[i] == 1 else 'blue'
        
        # Plot the line segment for the stroke
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], marker='o', color=color)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()



#------------------------------------------------------------------------------------------------------#


def build_intersection_matrix(node_features):
    num_strokes = node_features.shape[0]
    intersection_matrix = torch.zeros((num_strokes, num_strokes), dtype=torch.int32)

    for i in range(num_strokes):
        for j in range(i + 1, num_strokes):
            # Extract points for strokes i and j
            stroke_i_points = node_features[i].view(2, 3)
            stroke_j_points = node_features[j].view(2, 3)
            
            # Check for intersection
            if (stroke_i_points[0] == stroke_j_points[0]).all() or \
               (stroke_i_points[0] == stroke_j_points[1]).all() or \
               (stroke_i_points[1] == stroke_j_points[0]).all() or \
               (stroke_i_points[1] == stroke_j_points[1]).all():
                intersection_matrix[i, j] = 1
                intersection_matrix[j, i] = 1

    return intersection_matrix
