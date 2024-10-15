import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def get_kth_operation(op_to_index_matrix, k):    
    squeezed_matrix = op_to_index_matrix.squeeze(0)
    kth_operation = squeezed_matrix[:, k].unsqueeze(1)

    return kth_operation



def get_all_operation_strokes(stroke_operations_order_matrix, program_whole, operation):
    # Find the indices in program_whole where the operation occurs
    ks = [i for i, op in enumerate(program_whole) if op == operation]

    # Check if we found any valid indices
    if len(ks) == 0:
        raise ValueError(f"Operation '{operation}' not found in program_whole.")

    # Squeeze the matrix to remove any singleton dimensions at the 0-th axis
    squeezed_matrix = stroke_operations_order_matrix.squeeze(0)

    # Initialize an empty list to collect all columns corresponding to ks
    operation_strokes_list = []

    # Collect the k-th columns and stack them into a new matrix
    for k in ks:
        kth_operation = squeezed_matrix[:, k].unsqueeze(1)  # Extract and unsqueeze the k-th column
        operation_strokes_list.append(kth_operation)

    # Stack all the k-th columns into a matrix of shape (op_to_index.shape[0], n)
    all_operation_strokes = torch.cat(operation_strokes_list, dim=1)

    # Perform a logical OR (any row that has a 1 in any column will have 1 in the result)
    result_strokes = (all_operation_strokes > 0).any(dim=1).float().unsqueeze(1)

    # Return the result as a column vector of shape (op_to_index.shape[0], 1)
    return result_strokes


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
    def is_on_circle(point, center, radius, tolerance=0.05):
        distance = np.linalg.norm(point - center)
        return abs(distance - radius) < tolerance

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

                    if sketch_points[-1] != 0:
                        # the sketch is a circle:
                        center = sketch_points[:3]
                        radius = sketch_points[-1]
                        if is_on_circle(stroke_points[:3], center, radius) or is_on_circle(stroke_points[3:6], center, radius):
                            chosen = True
                            break

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


def program_mapping(program):
    operation_map = {
        'sketch': 1,
        'extrude': 2,
        'terminate': 0,
        'padding': 10
    }
    
    # Map each operation in the program list to its corresponding value
    mapped_program = [operation_map.get(op, -1) for op in program] 
    
    for i in range (20 - len(mapped_program)):
        mapped_program.append(10)
    
    return mapped_program


def program_gt_mapping(program):
    operation_map = {
        'sketch': 1,
        'extrude': 2,
        'terminate': 0,
        'padding': 10
    }
    
    # Map each operation in the program list to its corresponding value
    mapped_program = [operation_map.get(op, -1) for op in program]
    
    return mapped_program

#------------------------------------------------------------------------------------------------------#

def find_edge_features_slice(tensor, i):
    """
    Extract edges from the tensor where both nodes are within the range
    [i * 200, (i + 1) * 200), and adjust the values using modulo 200.

    Args:
    tensor (torch.Tensor): Input tensor of shape (2, n), where each column represents an edge between two nodes.
    i (int): The batch index.

    Returns:
    torch.Tensor: Filtered and adjusted tensor of shape (2, k) where both nodes in each edge are within [i * 200, (i + 1) * 200),
                  adjusted to range [0, 199] via modulo operation.
    """
    # Define the start and end of the range based on i
    start = i * 200
    end = (i + 1) * 200
    
    # Get the two rows representing the edges
    edges = tensor
    
    # Create a mask where both nodes in each edge are within the range [start, end)
    mask = (edges[0] >= start) & (edges[0] < end) & (edges[1] >= start) & (edges[1] < end)
    
    # Apply the mask to filter the edges
    filtered_edges = edges[:, mask]
    
    # Adjust the values to range [0, 199] using modulo 200
    adjusted_edges = filtered_edges % 200
    
    return adjusted_edges


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




#------------------------------------------------------------------------------------------------------#
def vis_full_graph(gnn_graph):
    """
    Visualize all strokes in the graph, regardless of whether they are used (i.e., the 8th value is 0 or 1).
    
    Parameters:
    gnn_graph (SketchLoopGraph): A single graph object containing loops and strokes.
    """

    # Extract stroke features
    stroke_node_features = gnn_graph['stroke'].x.numpy()

    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)

    # Initialize min and max limits
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    # Plot all strokes, regardless of whether they are used
    for stroke in stroke_node_features:
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


def vis_left_graph(stroke_node_features):
    """
    Visualize the graph with loops and strokes in 3D space, including circles for strokes where stroke[7] != 0.
    
    Parameters:
    graph (SketchLoopGraph): A single graph object containing loops and strokes.
    selected_loop_idx (int): The index of the loop that is chosen to be highlighted (not used yet).
    """

    # Extract stroke features
    stroke_node_features = feature_depad(stroke_node_features)

    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)

    # Initialize min and max limits
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    # Plot strokes based on the last value: 0 for blue, otherwise green
    for stroke in stroke_node_features:
        start, end = stroke[:3], stroke[3:6]
        stroke_color = 'blue' if stroke[8] == 0 else 'green'  # Color based on the last value

        # Update the min and max limits for rescaling based only on strokes (ignoring circles)
        if stroke[7] == 0:
            x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
            y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
            z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])

        if stroke[7] != 0:
            # Circle face (plot in green)
            x_values, y_values, z_values = plot_circle(stroke)

            ax.plot(x_values, y_values, z_values, color=stroke_color)

        else:
            # Plot the stroke (in blue or green depending on the condition)
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=stroke_color, linewidth=1)

    # Compute the center of the shape based on the strokes only (ignoring circles)
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
    Visualize the brep strokes and circular/cylindrical faces in 3D space if brep is not empty.
    
    Parameters:
    brep (np.ndarray or torch.Tensor): A matrix with shape (num_strokes, 6) representing strokes.
                       Each row contains two 3D points representing the start and end of a stroke.
                       If brep.shape[0] == 0, the function returns without plotting.
    """
    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)

    # Check if brep is empty
    if brep.shape[0] == 0:
        plt.title('Empty Plot')
        plt.show()
        return

    # Convert brep to numpy if it's a tensor
    if not isinstance(brep, np.ndarray):
        brep = brep.numpy()

    # Initialize min and max limits
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    # Plot all brep strokes and circle/cylinder faces in blue
    for stroke in brep:
        if stroke[6] != 0 and stroke[7] != 0:
            # Cylinder face
            center = stroke[:3]
            normal = stroke[3:6]
            height = stroke[6]
            radius = stroke[7]

            # Generate points for the cylinder's base circle (less dense)
            theta = np.linspace(0, 2 * np.pi, 30)  # Less dense with 30 points
            x_values = radius * np.cos(theta)
            y_values = radius * np.sin(theta)
            z_values = np.zeros_like(theta)

            # Combine the coordinates into a matrix (3, 30)
            base_circle_points = np.array([x_values, y_values, z_values])

            # Normalize the normal vector
            normal = normal / np.linalg.norm(normal)

            # Rotation logic using Rodrigues' formula
            z_axis = np.array([0, 0, 1])  # Z-axis is the default normal for the cylinder

            # Rotate the base circle points to align with the normal vector (even if normal is aligned)
            rotation_axis = np.cross(z_axis, normal)
            if np.linalg.norm(rotation_axis) > 0:  # Check if rotation is needed
                rotation_axis /= np.linalg.norm(rotation_axis)
                angle = np.arccos(np.clip(np.dot(z_axis, normal), -1.0, 1.0))

                # Create the rotation matrix using the rotation axis and angle (Rodrigues' rotation formula)
                K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                              [rotation_axis[2], 0, -rotation_axis[0]],
                              [-rotation_axis[1], rotation_axis[0], 0]])

                R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

                # Rotate the base circle points
                rotated_base_circle_points = np.dot(R, base_circle_points)
            else:
                rotated_base_circle_points = base_circle_points

            # Translate the base circle to the center point
            x_base = rotated_base_circle_points[0] + center[0]
            y_base = rotated_base_circle_points[1] + center[1]
            z_base = rotated_base_circle_points[2] + center[2]

            # Plot the base circle
            ax.plot(x_base, y_base, z_base, color='blue')

            # Plot vertical lines to create the "cylinder" (but without filling the body)
            x_top = x_base - normal[0] * height
            y_top = y_base - normal[1] * height
            z_top = z_base - normal[2] * height

            # Plot lines connecting the base and top circle with reduced density
            for i in range(0, len(x_base), 3):  # Fewer lines by skipping points
                ax.plot([x_base[i], x_top[i]], [y_base[i], y_top[i]], [z_base[i], z_top[i]], color='blue')

            # Update axis limits for the cylinder points
            x_min, x_max = min(x_min, x_base.min(), x_top.min()), max(x_max, x_base.max(), x_top.max())
            y_min, y_max = min(y_min, y_base.min(), y_top.min()), max(y_max, y_base.max(), y_top.max())
            z_min, z_max = min(z_min, z_base.min(), z_top.min()), max(z_max, z_base.max(), z_top.max())

        elif stroke[6] == 0 and stroke[7] != 0:
            # Circle face (same rotation logic as shared)
            x_values, y_values, z_values = plot_circle(stroke)

            ax.plot(x_values, y_values, z_values, color='blue')
        else:
            # Plot the stroke
            start, end = stroke[:3], stroke[3:6]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='blue', linewidth=1)

            # Update axis limits for the stroke points
            x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
            y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
            z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])

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


def vis_selected_loops(stroke_node_features, strokes_to_loops, selected_loop_idx):
    """
    Visualize the graph with loops and strokes in 3D space, including circles for strokes where stroke[7] != 0.
    
    Parameters:
    graph (SketchLoopGraph): A single graph object containing loops and strokes.
    selected_loop_idx (int): The index of the loop that is chosen to be highlighted (not used yet).
    """

    # Extract stroke features
    # stroke_node_features = graph['stroke'].x.cpu().numpy()
    stroke_node_features = feature_depad(stroke_node_features)

    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)

    # Initialize min and max limits
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    # Plot all strokes in blue
    for stroke in stroke_node_features:
        start, end = stroke[:3], stroke[3:6]

        # Update the min and max limits for rescaling based only on strokes (ignoring circles)
        if stroke[7] == 0:
            x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
            y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
            z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])
        
        if stroke[7] != 0:
            # Circle face
            x_values, y_values, z_values = plot_circle(stroke)

            ax.plot(x_values, y_values, z_values, color='blue')

        else:
            # Plot the stroke
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='blue', linewidth=1)


    # Plot the chosen loop in red
    # strokes_to_loops = graph['stroke', 'represents', 'loop'].edge_index
    stroke_to_loops = {}
    for stroke_idx, loop_idx in zip(strokes_to_loops[0], strokes_to_loops[1]):
        if loop_idx.item() not in stroke_to_loops:
            stroke_to_loops[loop_idx.item()] = []
        stroke_to_loops[loop_idx.item()].append(stroke_idx.item())

    for loop_idx, stroke_indices in stroke_to_loops.items():
        if loop_idx in selected_loop_idx:  # Plot only the selected loop
            for idx in stroke_indices:
                stroke = stroke_node_features[idx]
                if stroke[7] != 0:
                    x_values, y_values, z_values = plot_circle(stroke)
                    ax.plot(x_values, y_values, z_values, color='red')
                else:
                    start, end = stroke[:3], stroke[3:6]
                    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='red', linewidth=1)


    # Compute the center of the shape based on the strokes only (ignoring circles)
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



def vis_selected_strokes(stroke_node_features, selected_stroke_idx):
    """
    Visualizes selected strokes in 3D space.

    Parameters:
    - stroke_node_features: A numpy array or list containing the features of each stroke.
      Each stroke should contain its start and end coordinates, and potentially a flag indicating if it's a circle.
    - selected_stroke_idx: A list or array of indices of the strokes that should be highlighted in the visualization.
    
    This function visualizes all strokes but highlights the selected strokes.
    """
    
    # Extract stroke features
    # stroke_node_features = graph['stroke'].x.cpu().numpy()
    stroke_node_features = feature_depad(stroke_node_features)

    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)

    # Initialize min and max limits
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    # Plot all strokes in blue
    for stroke in stroke_node_features:
        start, end = stroke[:3], stroke[3:6]

        # Update the min and max limits for rescaling based only on strokes (ignoring circles)
        if stroke[7] == 0:
            x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
            y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
            z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])
        
        if stroke[7] != 0:
            # Circle face
            x_values, y_values, z_values = plot_circle(stroke)

            ax.plot(x_values, y_values, z_values, color='blue')

        else:
            # Plot the stroke
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='blue', linewidth=1)


    # Plot the chosen loop in red
    for idx, stroke in enumerate(stroke_node_features):
        if idx in selected_stroke_idx:
            stroke = stroke_node_features[idx]
            if stroke[7] != 0:
                x_values, y_values, z_values = plot_circle(stroke)
                ax.plot(x_values, y_values, z_values, color='red')
            else:
                start, end = stroke[:3], stroke[3:6]
                ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='red', linewidth=1)


    # Compute the center of the shape based on the strokes only (ignoring circles)
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



def feature_depad(feature_matrix):
    # Loop through each row and check if it's all -1
    for i, row in enumerate(feature_matrix):
        if np.all(row == -1):  # If all elements of the row are -1
            return feature_matrix[:i, :]  # Return the sub-matrix up to this row
    
    # If no row is full of -1, return the full matrix
    return feature_matrix


def plot_circle(stroke):
    center = stroke[:3]
    normal = stroke[3:6]
    radius = stroke[7]

    # Generate circle points in the XY plane
    theta = np.linspace(0, 2 * np.pi, 30)  # Less dense with 30 points
    x_values = radius * np.cos(theta)
    y_values = radius * np.sin(theta)
    z_values = np.zeros_like(theta)

    # Combine the coordinates into a matrix (3, 30)
    circle_points = np.array([x_values, y_values, z_values])

    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)

    # Rotation logic using Rodrigues' formula
    z_axis = np.array([0, 0, 1])  # Z-axis is the default normal for the circle

    rotation_axis = np.cross(z_axis, normal)
    if np.linalg.norm(rotation_axis) > 0:  # Check if rotation is needed
        rotation_axis /= np.linalg.norm(rotation_axis)
        angle = np.arccos(np.clip(np.dot(z_axis, normal), -1.0, 1.0))

        # Create the rotation matrix using the rotation axis and angle (Rodrigues' rotation formula)
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                        [rotation_axis[2], 0, -rotation_axis[0]],
                        [-rotation_axis[1], rotation_axis[0], 0]])

        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

        # Rotate the circle points
        rotated_circle_points = np.dot(R, circle_points)
    else:
        rotated_circle_points = circle_points

    # Translate the circle to the center point
    x_values = rotated_circle_points[0] + center[0]
    y_values = rotated_circle_points[1] + center[1]
    z_values = rotated_circle_points[2] + center[2]


    return x_values, y_values, z_values

