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

def vis_left_graph(graph):
    """
    Visualize only the strokes that are not used (i.e., have the 8th value as 0) in 3D space.
    
    Parameters:
    graph (SketchLoopGraph): A single graph object containing loops and strokes.
    """

    # Extract stroke features
    stroke_node_features = graph['stroke'].x.numpy()

    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)

    # Initialize min and max limits
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    # Plot only strokes where the 8th value (index 7) is 0
    for stroke in stroke_node_features:
        if stroke[7] == 0:  # Check if the 8th value is 0
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
            for i in range(0, len(x_base), 5):  # Fewer lines by skipping points
                ax.plot([x_base[i], x_top[i]], [y_base[i], y_top[i]], [z_base[i], z_top[i]], color='blue')

            # Update axis limits for the cylinder points
            x_min, x_max = min(x_min, x_base.min(), x_top.min()), max(x_max, x_base.max(), x_top.max())
            y_min, y_max = min(y_min, y_base.min(), y_top.min()), max(y_max, y_base.max(), y_top.max())
            z_min, z_max = min(z_min, z_base.min(), z_top.min()), max(z_max, z_base.max(), z_top.max())

        elif stroke[6] == 0 and stroke[7] != 0:
            # Circle face (same rotation logic as shared)
            center = stroke[:3]
            normal = stroke[3:6]
            radius = stroke[7]

            # Generate circle points in the XY plane
            theta = np.linspace(0, 2 * np.pi, 100)
            x_values = radius * np.cos(theta)
            y_values = radius * np.sin(theta)
            z_values = np.zeros_like(theta)

            # Combine the coordinates into a matrix (3, 100)
            circle_points = np.array([x_values, y_values, z_values])

            # Normalize the normal vector
            normal = normal / np.linalg.norm(normal)

            # Rotation logic using Rodrigues' formula
            z_axis = np.array([0, 0, 1])  # Z-axis is the default normal for the circle

            if not np.allclose(normal, z_axis):
                rotation_axis = np.cross(z_axis, normal)
                rotation_axis /= np.linalg.norm(rotation_axis)  # Normalize rotation axis
                angle = np.arccos(np.dot(z_axis, normal))

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

            # Plot the circle
            ax.plot(x_values, y_values, z_values, color='blue')

            # Update axis limits for the circle points
            x_min, x_max = min(x_min, x_values.min()), max(x_max, x_values.max())
            y_min, y_max = min(y_min, y_values.min()), max(y_max, y_values.max())
            z_min, z_max = min(z_min, z_values.min()), max(z_max, z_values.max())

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


def vis_selected_loops(graph, selected_loop):
    """
    Visualize the graph with loops and strokes in 3D space, including circles for strokes where stroke[7] != 0.
    
    Parameters:
    graph (SketchLoopGraph): A single graph object containing loops and strokes.
    """

    # Extract stroke features
    stroke_node_features = graph['stroke'].x.numpy()
    print("stroke_node_features", stroke_node_features)

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
        
        # Update the min and max limits for each axis
        x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
        y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
        z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])
        
        if stroke[7] != 0:
            print("stroke", stroke)
            # Circle face
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

            # Plot the circle
            ax.plot(x_values, y_values, z_values, color='blue')

            # Update axis limits for the circle points
            x_min, x_max = min(x_min, x_values.min()), max(x_max, x_values.max())
            y_min, y_max = min(y_min, y_values.min()), max(y_max, y_values.max())
            z_min, z_max = min(z_min, z_values.min()), max(z_max, z_values.max())

        else:
            # Plot the stroke
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


def vis_selected_strokes(graph, stroke_selection_mask):
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
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color, linewidth=1, alpha=1)


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
