import torch

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
