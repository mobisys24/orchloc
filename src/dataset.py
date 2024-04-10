import torch
from torch.utils.data import Dataset
import yaml
import argparse
import numpy as np
import os
import sys
from scipy.stats import truncnorm
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from collections import Counter


class ComplexDataset(Dataset):

    def __init__(self, 
                 amplitude_list, 
                 phase_list, 
                 labels):
        assert len(amplitude_list) == len(phase_list) == len(labels), "Mismatch in list lengths"
        self.amplitude_list = amplitude_list
        self.phase_list = phase_list
        self.labels = labels


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        amplitude = self.amplitude_list[idx]
        phase = self.phase_list[idx]
        label = self.labels[idx]
        real = amplitude * torch.cos(phase)
        imaginary = amplitude * torch.sin(phase)
        
        # stacks the real and imaginary tensors along a new dimension, 
        # resulting in a new tensor with shape (2, 16).
        complex_number = torch.stack((real, imaginary), dim=0)
        return complex_number, label


class ComplexDatasetLocs(Dataset):

    def __init__(self, 
                 amplitude_list, 
                 phase_list, 
                 labels, 
                 label_feature_path, 
                 row_col_idx_dict):
        assert len(amplitude_list) == len(phase_list) == len(labels), "Mismatch in list lengths"

        self.amplitude_list = amplitude_list
        self.phase_list = phase_list
        self.labels = labels
        self.label_features = self.load_label_features(label_feature_path, row_col_idx_dict)

        unique_ids = torch.unique(self.labels)
        if len(self.label_features) != len(unique_ids):
            raise ValueError(f"Mismatch between label_features and labels. Only {len(self.label_features)} location feature were obtained, but there are a total of {len(unique_ids)} location IDs. \n\nPlease make sure to finish running '0_location_representation.py' before running this script.\n")        


    def load_label_features(self, path, row_col_idx_dict):

        label_features = {}
        with open(path, 'r') as f:

            for line in f:
                parts = line.strip().split(',')
                label_id = parts[0]  # '1_1', '1_2', etc.

                vector = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)

                if label_id in row_col_idx_dict:
                    mapped_id = row_col_idx_dict[label_id]
                    label_features[mapped_id] = vector

        return label_features


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        amplitude = self.amplitude_list[idx]
        phase = self.phase_list[idx]
        label = self.labels[idx]
        real = amplitude * torch.cos(phase)
        imaginary = amplitude * torch.sin(phase)

        complex_number = torch.stack((real, imaginary), dim=0)

        label_feature = self.label_features.get(label.item())

        return complex_number, label_feature, label


class ShiftedDataset(Dataset):

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset


    def __getitem__(self, idx):
        data, label, label_int = self.base_dataset[idx]

        return data, label, label_int + 1   # Shift labels from starting at 0 to start at 1


    def __len__(self):
        return len(self.base_dataset)


class ComplexDataset_real_imagary_v2(Dataset):
    
    def __init__(self, 
                 features, 
                 features_label, 
                 labels, 
                 labels_int):
        assert len(features) == len(features_label) == len(labels) == len(labels_int), "Mismatch in list lengths"
        self.features = features
        self.features_label = features_label 
        self.labels = labels
        self.labels_int = labels_int


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        label = self.labels[idx]
        label_int = self.labels_int[idx]

        complex_number = self.features[idx]
        complex_number_label = self.features_label[idx]

        return complex_number, complex_number_label, label, label_int


def generate_three_loader_v2(dataset_t, 
                             n_batch_size, 
                             valid_size, 
                             test_size):
    
    labels = [label for _, label in dataset_t]
    
    indices = np.arange(len(dataset_t))
    
    train_temp_indices, test_indices = train_test_split(
        indices, test_size=test_size, stratify=labels, random_state=42)
    
    train_temp_labels = [labels[i] for i in train_temp_indices]
    
    train_indices, valid_indices = train_test_split(
        train_temp_indices, test_size=valid_size, stratify=train_temp_labels, random_state=42)
    
    # Create dataset subsets for each split
    train_dataset = Subset(dataset_t, train_indices)
    valid_dataset = Subset(dataset_t, valid_indices)
    test_dataset = Subset(dataset_t, test_indices)
    
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=n_batch_size, shuffle=True)
    valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=len(valid_indices), shuffle=False)
    test_data_loader  = DataLoader(dataset=test_dataset, batch_size=len(test_indices), shuffle=False)
    
    print(f'Loaded {len(dataset_t)} samples, split {len(train_indices)}/{len(valid_indices)}/{len(test_indices)} for train/valid/test.')
    
    return train_data_loader, valid_data_loader, test_data_loader


def generate_three_loader_v3(dataset_t, 
                             n_batch_size, 
                             valid_size, 
                             test_size):
    labels = [label for _, _, _, label in dataset_t]
    
    indices = np.arange(len(dataset_t))
    
    train_temp_indices, test_indices = train_test_split(
        indices, test_size=test_size, stratify=labels, random_state=42)
    
    train_temp_labels = [labels[i] for i in train_temp_indices]
    
    train_indices, valid_indices = train_test_split(
        train_temp_indices, test_size=valid_size, stratify=train_temp_labels, random_state=42)
    
    # Create dataset subsets for each split
    train_dataset = Subset(dataset_t, train_indices)
    valid_dataset = Subset(dataset_t, valid_indices)
    test_dataset = Subset(dataset_t, test_indices)
    
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=n_batch_size, shuffle=True)
    valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=len(valid_indices), shuffle=False)
    test_data_loader  = DataLoader(dataset=test_dataset, batch_size=len(test_indices), shuffle=False)
    
    print(f'Loaded {len(dataset_t)} samples, split {len(train_indices)}/{len(valid_indices)}/{len(test_indices)} for train/valid/test.')
    
    return train_data_loader, valid_data_loader, test_data_loader


def generate_three_dataset_v2(dataset_t, 
                              valid_size, 
                              test_size):
    
    labels = [label for _, _, label in dataset_t]
    
    indices = np.arange(len(dataset_t))
    
    train_temp_indices, test_indices = train_test_split(
        indices, test_size=test_size, stratify=labels, random_state=42)
    
    train_temp_labels = [labels[i] for i in train_temp_indices]
    
    train_indices, valid_indices = train_test_split(
        train_temp_indices, test_size=valid_size, stratify=train_temp_labels, random_state=42)
    
    train_dataset = Subset(dataset_t, train_indices)
    valid_dataset = Subset(dataset_t, valid_indices)
    test_dataset = Subset(dataset_t, test_indices)

    print(f'Loaded {len(dataset_t)} samples, split {len(train_indices)}/{len(valid_indices)}/{len(test_indices)} for train/valid/test.')
    
    return train_dataset, valid_dataset, test_dataset


def generate_three_dataset_v3(dataset_t, 
                              ratios, 
                              valid_r, 
                              test_r):

    # Assuming dataset_t.labels is a list or tensor of labels
    labels = np.array([label for _, _, label in dataset_t])
    unique_labels = np.unique(labels)
    num_selected_labels = int(len(unique_labels) * ratios)
    selected_label_ids = np.random.choice(unique_labels, num_selected_labels, replace=False)

    # Filter indices for selected label IDs
    filtered_indices = [i for i, label in enumerate(labels) if label in selected_label_ids]
    filtered_labels = labels[filtered_indices]

    # Convert filtered indices to np.array for compatibility with train_test_split stratify parameter
    filtered_indices = np.array(filtered_indices)
    
    # Calculate actual sizes for validation and test sets based on the filtered dataset
    total_size = len(filtered_indices)
    test_size = int(total_size * test_r)
    valid_size = int(total_size * valid_r)

    # Stratified split to ensure equal representation of each label ID
    train_temp_indices, test_indices = train_test_split(
        filtered_indices, test_size=test_size, stratify=filtered_labels, random_state=42)
    
    # Update labels for stratified split for validation
    train_temp_labels = filtered_labels[[np.where(filtered_indices == i)[0][0] for i in train_temp_indices]]
    
    train_indices, valid_indices = train_test_split(
        train_temp_indices, test_size=valid_size, stratify=train_temp_labels, random_state=42)

    # Create subsets for train, validation, and test datasets
    train_dataset = Subset(dataset_t, train_indices)
    valid_dataset = Subset(dataset_t, valid_indices)
    test_dataset = Subset(dataset_t, test_indices)

    return train_dataset, valid_dataset, test_dataset


def generate_three_dataset_v4(dataset_t, 
                              ratios, 
                              valid_r, 
                              test_r):
    
    labels = np.array([label for _, _, label in dataset_t])
    unique_labels = np.unique(labels)

    # Calculate the number of labels to select based on ratios
    num_selected_labels = int(len(unique_labels) * ratios)

    # Randomly select label IDs for the first part
    selected_label_ids = np.random.choice(unique_labels, num_selected_labels, replace=False)

    # Identify remaining label IDs for the second part
    remaining_label_ids = np.setdiff1d(unique_labels, selected_label_ids)

    # Helper function to split dataset based on label IDs
    def split_dataset_by_labels(label_ids):
        filtered_indices = [i for i, label in enumerate(labels) if label in label_ids]
        filtered_labels = labels[filtered_indices]

        # Calculate actual sizes for validation and test sets based on the filtered dataset
        total_size = len(filtered_indices)
        test_size = int(total_size * test_r)
        valid_size = int(total_size * valid_r)

        # Stratified split to ensure equal representation of each label ID
        train_temp_indices, test_indices = train_test_split(
            filtered_indices, test_size=test_size, stratify=filtered_labels, random_state=42)
        
        train_temp_labels = filtered_labels[[np.where(filtered_indices == i)[0][0] for i in train_temp_indices]]
        
        train_indices, valid_indices = train_test_split(
            train_temp_indices, test_size=valid_size, stratify=train_temp_labels, random_state=42)

        # Create subsets for train, validation, and test datasets
        train_dataset = Subset(dataset_t, train_indices)
        valid_dataset = Subset(dataset_t, valid_indices)
        test_dataset = Subset(dataset_t, test_indices)

        return train_dataset, valid_dataset, test_dataset

    # Split datasets for selected and remaining label IDs
    train_dataset_1, valid_dataset_1, test_dataset_1 = split_dataset_by_labels(selected_label_ids)
    train_dataset_2, valid_dataset_2, test_dataset_2 = split_dataset_by_labels(remaining_label_ids)


    return train_dataset_1, valid_dataset_1, test_dataset_1, train_dataset_2, valid_dataset_2, test_dataset_2



