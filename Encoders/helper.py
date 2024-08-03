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

    
#------------------------------------------------------------------------------------------------------#
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
