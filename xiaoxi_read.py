
import Preprocessing.dataloader
import tdqm

dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/cad2sketch_annotated')
for data in tqdm(dataset, desc=f"xiaoxixiaoxi"):
    data_idx, program, program_whole, stroke_cloud_loops, stroke_node_features, strokes_perpendicular, output_brep_edges, stroke_operations_order_matrix, loop_neighboring_vertical, loop_neighboring_horizontal,loop_neighboring_contained, stroke_to_loop, stroke_to_edge = data

    # stroke_node_features is what you want





# this is how to vis stroke_node_features



# Straight Line: 10 values + type 1
# 0-2: point1, 3-5:point2, 6:alpha_value, 7-9: 0

# Circle Feature: 10 values + type 2
# 0-2: center, 3-5:normal, 6:alpha_value, 7:radius, 8-9: 0

# Arc Feature: 10 values + type 3
# 0-2: point1, 3-5:point2, 6:alpha_value, 7-9:center

# Ellipse Feature: 10 values + type 4
# 0-2: center, 3-5:normal, 6:alpha_value, 7: major axis, 8: minor axis, 9: orientation

# Closed Line: 10 values + type 5
# 0-2: point1, 3-5: point2, 6:alpha_value, 7-9: random point in the line

# Curved Line: 10 values + type 6
# 0-2: point1, 3-5: point2, 6:alpha_value, 7-9: random point in the line


def vis_selected_strokes(stroke_node_features, selected_stroke_idx, data_idx, alpha_value=0.7):
    """
    Visualizes selected strokes in 3D space with a hand-drawn effect.

    Parameters:
    - stroke_node_features: A numpy array or list containing the features of each stroke.
    - selected_stroke_idx: A list or array of indices of the strokes that should be highlighted in red.
    - data_idx: string to locate the dataset subfolder
    - alpha_value: Float, default 0.7, controls transparency of red highlighted strokes.
    """
    import os
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import random

    # Load the already-processed all_lines
    final_edges_file_path = os.path.join(
        os.getcwd(), 'dataset', 'cad2sketch_annotated', data_idx, 'perturbed_all_lines.json')
    all_lines = read_json(final_edges_file_path)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # === Set Camera View ===
    # ax.view_init(elev=-150, azim=57, roll=0)  # Match the provided camera inclination and azimuth
    # ax.dist = 7  # Simulate distance/zoom, optional

    # Clean plot styling
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_frame_on(False)
    ax.grid(False)
    ax.set_axis_off()

    # Bounding box init
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    # === First pass: plot all strokes ===
    for i, stroke in enumerate(all_lines):
        geometry = stroke["geometry"]
        if len(geometry) < 2:
            continue

        alpha = stroke["opacity"]

        for j in range(1, len(geometry)):
            start = geometry[j - 1]
            end = geometry[j]

            x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
            y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
            z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])

            ax.plot([start[0], end[0]],
                    [start[1], end[1]],
                    [start[2], end[2]],
                    color='black',
                    linewidth=0.6,
                    alpha=alpha)

    # === Rescale view ===
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)

    ax.set_xlim([x_center - max_diff / 2, x_center + max_diff / 2])
    ax.set_ylim([y_center - max_diff / 2, y_center + max_diff / 2])
    ax.set_zlim([z_center - max_diff / 2, z_center + max_diff / 2])

    # === Second pass: highlight selected strokes in red ===
    for idx in selected_stroke_idx:
        if idx < len(all_lines):
            geometry = all_lines[idx]["geometry"]
            if len(geometry) < 2:
                continue
            for j in range(1, len(geometry)):
                start = geometry[j - 1]
                end = geometry[j]
                ax.plot([start[0], end[0]],
                        [start[1], end[1]],
                        [start[2], end[2]],
                        color='red',
                        linewidth=2.0,
                        alpha=alpha_value)
        else:
            stroke = stroke_node_features[idx]
            start = stroke[0:3]
            end = stroke[3:6]
            ax.plot([start[0], end[0]],
                    [start[1], end[1]],
                    [start[2], end[2]],
                    color='red',
                    linewidth=2.0,
                    alpha=alpha_value)

    plt.show()




