import struct
import os
import numpy as np
import glob
from scipy.linalg import sqrtm
import torch


def io_read_iq(filepath):

	binfile = open(filepath, 'rb')

	size = os.path.getsize(filepath)
	bytes_one_sample = 4
	len_chirp = int(size / bytes_one_sample)

	one_chirp_sample_list = []
	for _ in range(len_chirp):
		data = binfile.read(bytes_one_sample)
		num = struct.unpack('f', data)
		one_chirp_sample_list.append(num[0])

	binfile.close()

	iq_stream = []
	for idx in range(int(len_chirp / 2)):

		data_i = one_chirp_sample_list[idx * 2]
		data_q = one_chirp_sample_list[idx * 2 + 1]
		complex_data = complex(data_i, data_q)
		iq_stream.append(complex_data)

	return iq_stream


def get_newest_model_path(prefix, folder_path):
	files = glob.glob(os.path.join(folder_path, prefix))

	newest_file = max(files, key=os.path.getctime)

	return newest_file


def get_newest_model_path_epoch(prefix, folder_path):
    files = glob.glob(os.path.join(folder_path, prefix))

    newest_file = max(files, key=lambda x: (os.path.getctime(x), int(os.path.basename(x).split('_')[1])))

    return newest_file


def compute_amplitude_3d(x):
	real_part = x[:, 0, :] ** 2
	imag_part = x[:, 1, :] ** 2

	abs_value = np.sqrt(real_part + imag_part)

	return abs_value


def get_features_by_label_v4(dataset, target_label):
	features_by_label = []
	label_vec_by_label = []
	label_int_by_label = []

	for i in range(len(dataset)):
		feature, label, label_int = dataset[i]

		if label_int == target_label:
			features_by_label.append(feature)
			label_vec_by_label.append(label)
			label_int_by_label.append(label_int)

	return torch.stack(features_by_label), torch.stack(label_vec_by_label), torch.stack(label_int_by_label)


def calculate_fid(real_features, generated_features):
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    return fid


def calculate_distance_error(labels_pred_d: torch.Tensor, 
                             labels_int_d: torch.Tensor, 
                             num_rows_d: int, 
                             num_cols_d: int, 
                             row_dist_d: float, 
                             col_dist_d: float) -> float:

    tree_id_to_loc = {(r * num_cols_d + c): (r, c) for r in range(num_rows_d) for c in range(num_cols_d)}

    loc_tensor = torch.tensor(list(tree_id_to_loc.values()), dtype=torch.float32)

    loc_pred = loc_tensor[labels_pred_d]
    loc_true = loc_tensor[labels_int_d]

    # Calculate the distances in the row and column dimensions
    dist_row = (loc_pred[:, 0] - loc_true[:, 0]) * row_dist_d
    dist_col = (loc_pred[:, 1] - loc_true[:, 1]) * col_dist_d

    # Calculate the Euclidean distance
    dist = torch.sqrt(dist_row ** 2 + dist_col ** 2)

    return dist


class OrchardDistance:
    def __init__(self, num_rows, num_cols, row_dist, col_dist, orchard_origin):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.row_dist = row_dist
        self.col_dist = col_dist
        self.orchard_origin = np.array(orchard_origin)

        self.trees = []
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                x = self.orchard_origin[0] + i * self.row_dist

                y = self.orchard_origin[1] + j * self.col_dist + (i % 2) * self.col_dist / 2
                position = (x, y)
                self.trees.append(position)

    def get_tree_position(self, index):
        return self.trees[index]


