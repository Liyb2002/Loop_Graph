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


def predict_face_coplanar_with_brep(predicted_index, coplanar_matrix, node_features):
    # Step 1: Find all coplanar faces with the predicted_index using coplanar_matrix
    coplanar_faces = torch.where(coplanar_matrix[predicted_index] == 1)[0]
    
    # Step 2: Check if any of the coplanar faces are used, using the last column of node_features
    if torch.any(node_features[coplanar_faces, -1] == 1):
        return True
    
    # Step 3: If no coplanar face is used, return False
    return False

#------------------------------------------------------------------------------------------------------#

def vis_stroke_cloud(node_features):
    if node_features.shape[1] == 0:
        return 

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    node_features = node_features.squeeze(0)

    # Initialize min and max limits
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    # Plot all strokes in blue and compute limits
    for stroke in node_features:
        start = stroke[:3].numpy()
        end = stroke[3:].numpy()

        # Update the min and max limits for each axis
        x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
        y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
        z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])

        # Plot the line segment for the stroke in blue
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], marker='o', color='blue')

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
    
    # Initialize min and max limits
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')
    
    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each stroke and compute limits
    for i, stroke in enumerate(node_features):
        x = [stroke[0], stroke[3]]
        y = [stroke[1], stroke[4]]
        z = [stroke[2], stroke[5]]
        
        # Update the min and max limits for each axis
        x_min, x_max = min(x_min, x[0], x[1]), max(x_max, x[0], x[1])
        y_min, y_max = min(y_min, y[0], y[1]), max(y_max, y[0], y[1])
        z_min, z_max = min(z_min, z[0], z[1]), max(z_max, z[0], z[1])
        
        ax.plot(x, y, z, color=stroke_colors[i])
    
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

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def vis_brep_and_nextSketch(matrix, face_to_stroke, node_features, edge_features):
    if edge_features.shape[1] == 0:
        return
    
    # Initialize a list to keep track of stroke colors
    stroke_colors = ['blue'] * node_features.shape[0]

    # Find the index of the item with the highest value in the matrix
    max_index = torch.argmax(matrix).item()

    # Set the strokes in the face with the highest value to red
    for stroke in face_to_stroke[max_index]:
        stroke_colors[stroke] = 'red'
    
    # Initialize min and max limits
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')
    
    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot node_features in blue (or red if in max_index face) and compute limits
    for i, stroke in enumerate(node_features):
        x = [stroke[0], stroke[3]]
        y = [stroke[1], stroke[4]]
        z = [stroke[2], stroke[5]]

        # Update the min and max limits for each axis
        x_min, x_max = min(x_min, x[0], x[1]), max(x_max, x[0], x[1])
        y_min, y_max = min(y_min, y[0], y[1]), max(y_max, y[0], y[1])
        z_min, z_max = min(z_min, z[0], z[1]), max(z_max, z[0], z[1])

        ax.plot(x, y, z, color=stroke_colors[i])
    
    # Plot edge_features in green and compute limits
    for stroke in edge_features:
        x = [stroke[0], stroke[3]]
        y = [stroke[1], stroke[4]]
        z = [stroke[2], stroke[5]]

        # Update the min and max limits for each axis
        x_min, x_max = min(x_min, x[0], x[1]), max(x_max, x[0], x[1])
        y_min, y_max = min(y_min, y[0], y[1]), max(y_max, y[0], y[1])
        z_min, z_max = min(z_min, z[0], z[1]), max(z_max, z[0], z[1])

        ax.plot(x, y, z, color='green')

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

    # Plot the strokes in red (again for visibility after scaling)
    for i, stroke in enumerate(node_features):
        if stroke_colors[i] == 'red':
            x = [stroke[0], stroke[3]]
            y = [stroke[1], stroke[4]]
            z = [stroke[2], stroke[5]]
            ax.plot(x, y, z, color=stroke_colors[i])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()



def vis_clean_strokes(node_features, edge_features):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert tensors to set of tuples for easy comparison
    edge_set = {tuple(edge.numpy()) for edge in edge_features}

    # Initialize min and max limits
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    # Plot node_features in blue if they are not in edge_features
    for stroke in node_features:
        stroke_tuple = tuple(stroke.numpy())
        
        if stroke_tuple not in edge_set:
            x = [stroke[0], stroke[3]]
            y = [stroke[1], stroke[4]]
            z = [stroke[2], stroke[5]]

            # Update the min and max limits for each axis
            x_min, x_max = min(x_min, x[0], x[1]), max(x_max, x[0], x[1])
            y_min, y_max = min(y_min, y[0], y[1]), max(y_max, y[0], y[1])
            z_min, z_max = min(z_min, z[0], z[1]), max(z_max, z[0], z[1])

            ax.plot(x, y, z, color='blue')

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


#------------------------------------------------------------------------------------------------------#


def clean_face_choice(predicted_index, node_features):
    # Check if the predicted_index itself is not being used
    if node_features[predicted_index, -1] == 0:
        return True
    else:
        return False
