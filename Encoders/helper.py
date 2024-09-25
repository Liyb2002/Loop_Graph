import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def get_kth_operation(op_to_index_matrix, k):    
    squeezed_matrix = op_to_index_matrix.squeeze(0)
    kth_operation = squeezed_matrix[:, k].unsqueeze(1)

    return kth_operation


def choose_extrude_strokes(stroke_selection_mask, sketch_selection_mask, stroke_node_features):
    """
    Given stroke_selection_mask and sketch_selection_mask, find if a stroke in stroke_selection_mask
    has one point in common with a stroke in sketch_selection_mask and mark it as chosen.
    
    Parameters:
    stroke_selection_mask (np.ndarray): A binary mask of shape (num_strokes, 1) for extrude strokes.
    sketch_selection_mask (np.ndarray): A binary mask of shape (num_strokes, 1) for sketch strokes.
    stroke_node_features (np.ndarray): A numpy array of shape (num_strokes, 6), where each row contains two 3D points.
    
    Returns:
    extrude_strokes (np.ndarray): A binary mask of shape (num_strokes, 1), indicating which extrude strokes are chosen.
    """
    num_strokes = stroke_selection_mask.shape[0]
    
    # Initialize the output matrix with zeros
    extrude_strokes = torch.zeros((num_strokes, 1), dtype=torch.float32)
    
    # Iterate through all strokes in stroke_selection_mask
    for i in range(num_strokes):
        # If the stroke is marked in stroke_selection_mask
        if stroke_selection_mask[i] == 1:
            stroke_points = stroke_node_features[i]  # Get the 3D points of the stroke
            chosen = False

            # Check if any stroke in sketch_selection_mask shares a point with this stroke
            for j in range(num_strokes):
                if sketch_selection_mask[j] == 1:
                    sketch_points = stroke_node_features[j]  # Get the 3D points of the sketch stroke

                    # Compare points of the stroke with the points of the sketch stroke using np.allclose
                    if (np.allclose(stroke_points[:3], sketch_points[:3]) or np.allclose(stroke_points[:3], sketch_points[3:6]) or
                        np.allclose(stroke_points[3:6], sketch_points[:3]) or np.allclose(stroke_points[3:6], sketch_points[3:6])):
                        chosen = True
                        break

            # If the stroke has one of its points in any of the sketch strokes, mark it as chosen
            if chosen:
                extrude_strokes[i] = 1

    return extrude_strokes


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


def vis_specific_loop(loop, strokes):
    """
    Visualize specific loops and strokes.
    
    Parameters:
    loop (list of int): A list containing indices of the strokes to be highlighted.
    strokes (np.ndarray): A matrix of shape (num_strokes, 7), where the first 6 columns represent two 3D points.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot all strokes in blue
    for stroke in strokes:
        start, end = stroke[:3], stroke[3:6]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='blue', alpha=0.5)

    # Plot strokes in the loop in red
    for idx in loop:
        stroke = strokes[idx]
        start, end = stroke[:3], stroke[3:6]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='red', linewidth=2)

    # Set labels and show plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


#------------------------------------------------------------------------------------------------------#
def vis_partial_graph(loops, strokes, stroke_to_brep):
    """
    Visualize multiple loops and strokes in 3D space, with all strokes in blue and rescaled axes.
    Only visualize strokes that are part of a loop and exclude loops based on the stroke_to_brep matrix.

    Parameters:
    loops (list of lists of int): A list of loops, where each loop is a list containing indices of strokes to be highlighted.
    strokes (np.ndarray): A matrix of shape (num_strokes, 7), where the first 6 columns represent two 3D points.
    stroke_to_brep (torch.Tensor): A tensor that either has shape (num_loops, num_brep) or [0].
                                   If its shape is (num_loops, num_brep), a loop is excluded if it has a column value of 1.
    """
    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)

    # Initialize min and max limits
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    # Determine which loops to visualize based on stroke_to_brep
    if stroke_to_brep.shape[0] > 0:
        # Identify loops to exclude (any loop with a column value of 1)
        exclude_loops = [i for i in range(stroke_to_brep.shape[0]) if torch.any(stroke_to_brep[i] == 1)]
    else:
        exclude_loops = []

    # Plot strokes in each loop with a line width of 0.5
    for i, loop in enumerate(loops):
        if i not in exclude_loops:
            for idx in loop:
                stroke = strokes[idx]
                start, end = stroke[:3], stroke[3:6]

                # Update the min and max limits for each axis
                x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
                y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
                z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])
                
                ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='blue', linewidth=1)

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

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show plot
    plt.show()



def vis_brep(brep):
    """
    Visualize the brep strokes in 3D space if brep is not empty.
    
    Parameters:
    brep (np.ndarray): A matrix with shape (num_strokes, 6) representing strokes.
                       Each row contains two 3D points representing the start and end of a stroke.
                       If brep.shape[0] == 0, the function returns without plotting.
    """
    # Check if brep is empty
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)

    # Check if brep is empty
    if brep.shape[0] == 0:
        plt.title('Empty Plot')
        plt.show()
        return
    

    # Initialize min and max limits
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    # Plot all brep strokes in blue with line width 1
    for stroke in brep:
        start, end = stroke[:3], stroke[3:6]
        
        # Update the min and max limits for each axis
        x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
        y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
        z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])
        
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='blue', linewidth=1)

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

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show plot
    plt.show()


def vis_whole_graph(graph, selected_loop):
    """
    Visualize the graph with loops and strokes in 3D space.
    
    Parameters:
    graph (SketchLoopGraph): A single graph object containing loops and strokes.
    loop_selection_masks (np.ndarray or torch.Tensor): A binary mask of shape (num_loops, 1), where 1 indicates a chosen loop.
    """

    # Extract loop-stroke connection edges and stroke features
    loops_to_strokes = graph['loop', 'representedBy', 'stroke'].edge_index
    stroke_node_features = graph['stroke'].x.numpy()

    # Convert edge indices to a more accessible format
    loop_to_strokes = {}
    for loop_idx, stroke_idx in zip(loops_to_strokes[0], loops_to_strokes[1]):
        if loop_idx.item() not in loop_to_strokes:
            loop_to_strokes[loop_idx.item()] = []
        loop_to_strokes[loop_idx.item()].append(stroke_idx.item())

    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)

    # Initialize min and max limits
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    # Plot all loops in blue
    for loop_idx, stroke_indices in loop_to_strokes.items():
        for idx in stroke_indices:
            stroke = stroke_node_features[idx]
            start, end = stroke[:3], stroke[3:6]
            
            # Update the min and max limits for each axis
            x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
            y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
            z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])
            
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='blue', linewidth=1, alpha=0.5)

    # Plot chosen loops in red
    for loop_idx, stroke_indices in loop_to_strokes.items():
        if loop_idx == selected_loop:  # Plot only the selected loop
            for idx in stroke_indices:
                stroke = stroke_node_features[idx]
                start, end = stroke[:3], stroke[3:6]
                ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='red', linewidth=1)

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

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show plot
    plt.show()


def vis_used_graph(graph):
    """
    Visualize the graph with loops and strokes in 3D space.
    Only visualize the loops with feature 0 (blue loops).
    
    Parameters:
    graph (SketchLoopGraph): A single graph object containing loops and strokes.
    """

    # Extract loop-stroke connection edges and stroke features
    loops_to_strokes = graph['loop', 'representedBy', 'stroke'].edge_index
    stroke_node_features = graph['stroke'].x.numpy()
    loop_features = graph['loop'].x.numpy()  # Assuming graph['loop'].x contains the features for each loop

    # Convert edge indices to a more accessible format
    loop_to_strokes = {}
    for loop_idx, stroke_idx in zip(loops_to_strokes[0], loops_to_strokes[1]):
        if loop_idx.item() not in loop_to_strokes:
            loop_to_strokes[loop_idx.item()] = []
        loop_to_strokes[loop_idx.item()].append(stroke_idx.item())

    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)

    # Initialize min and max limits
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    # Plot only the unused loops (with feature 0)
    for loop_idx, stroke_indices in loop_to_strokes.items():
        # Check if the current loop has a feature of 0
        if loop_features[loop_idx][0] == 0:  # Only visualize if the first feature is 0
            for idx in stroke_indices:
                stroke = stroke_node_features[idx]
                start, end = stroke[:3], stroke[3:6]

                # Update the min and max limits for each axis
                x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
                y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
                z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])

                # Plot the stroke in blue
                ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='blue', linewidth=1, alpha=0.5)

    # Compute the center of the shape
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2

    # Compute the maximum difference across x, y, z directions
    max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)

    # Set the same limits for x, y, and z axes centered around the computed center
    if max_diff > 0:  # Only set limits if max_diff is positive
        ax.set_xlim([x_center - max_diff / 2, x_center + max_diff / 2])
        ax.set_ylim([y_center - max_diff / 2, y_center + max_diff / 2])
        ax.set_zlim([z_center - max_diff / 2, z_center + max_diff / 2])

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show plot
    plt.show()


def vis_stroke_graph(graph, stroke_selection_mask):
    """
    Visualize all strokes in the graph in 3D space. Strokes are colored based on stroke_selection_mask.
    
    Parameters:
    graph (SketchHeteroData): A single graph object containing strokes.
    stroke_selection_mask (np.ndarray or torch.Tensor): A binary mask of shape (num_strokes, 1), where 1 indicates a chosen stroke (red), and 0 indicates an unchosen stroke (blue).
    """

    # Extract stroke features from the graph
    stroke_node_features = graph['stroke'].x.numpy()
    stroke_selection_mask = stroke_selection_mask.numpy()

    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)

    # Initialize min and max limits
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    # Plot all strokes, using red for selected strokes and blue for unchosen ones
    for idx, stroke in enumerate(stroke_node_features):
        start, end = stroke[:3], stroke[3:6]
        color = 'red' if stroke_selection_mask[idx] > 0.5 else 'blue'
        
        # Update the min and max limits for each axis
        x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
        y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
        z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])
        
        # Plot the stroke
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color, linewidth=1, alpha=0.8 if color == 'red' else 0.5)

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

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show plot
    plt.show()


def vis_stroke_with_order(stroke_node_features):
    """
    Visualize strokes progressively. Initially plots 1 stroke, then 2 strokes, and so on.
    
    Parameters:
    stroke_node_features (np.ndarray): A matrix of shape (num_strokes, 6), where the first 3 columns 
                                       represent the start point and the next 3 columns represent the end point.
    """

    # Loop through the stroke_node_features progressively, plotting one more stroke each time
    for i in range(1, len(stroke_node_features) + 1):
        # Initialize the 3D plot for each step
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(False)

        # Plot the first i strokes
        for stroke in stroke_node_features[:i]:
            start, end = stroke[:3], stroke[3:6]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='blue', linewidth=1)

        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Set equal scaling for all axes
        ax.set_box_aspect([1, 1, 1])

        # Show plot for each step
        plt.show()
    