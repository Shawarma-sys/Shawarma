# *************************************************************************
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# *************************************************************************

import copy
import sys
import numpy as np
import pandas as pd
import os

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class UnifiedTrafficDataset(Dataset):
    def __init__(self, data_file, labels_file, sequence_length=None, standardize=False, normalize=False, device='cpu'):
        """
        Unified dataset for both flat (MLP) and sequence (CNN/RNN) traffic data.
        Automatically detects data dimension and applies appropriate processing.
        """
        if not os.path.isfile(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        if not os.path.isfile(labels_file):
            raise FileNotFoundError(f"Labels file not found: {labels_file}")

        # Load data
        data_np = np.load(data_file)
        self.labels = torch.from_numpy(np.load(labels_file)).long()

        if data_np.ndim == 3:
            # Sequence data: (samples, sequence_length, features)
            self.is_sequence = True
            self.features = torch.from_numpy(data_np).long()
            if sequence_length and self.features.shape[1] != sequence_length:
                raise ValueError(f"Sequence length mismatch: expected {sequence_length}, got {self.features.shape[1]}")
        else:
            # Flat data: (samples, features)
            self.is_sequence = False
            self.features = torch.from_numpy(data_np).float()
            
            if standardize:
                features_mean = torch.mean(self.features, dim=0).unsqueeze(0)
                features_std = torch.std(self.features, dim=0).unsqueeze(0)
                self.features = torch.nan_to_num((self.features - features_mean) / features_std)

            if normalize:
                scaler = MinMaxScaler()
                self.features = torch.from_numpy(scaler.fit_transform(self.features.numpy())).float()

        # Move to device
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def traffic_collate_fn(batch):
    """
    Standard collate function for UnifiedTrafficDataset.
    """
    features, labels = zip(*batch)
    features_stacked = torch.stack(features, dim=0)
    labels_stacked = torch.stack(labels, dim=0)
    return features_stacked, labels_stacked.long()


class OnlineDataset(Dataset):
    def __init__(self, features, labels, standardize=True, normalize=False, device='cuda'):
        self.features = features.type(torch.float32)
        self.labels = labels.type(torch.int64)
        # data pre-processing
        ## standardize the data
        if standardize:
            features_mean = torch.mean(self.features, dim=0).unsqueeze(0)
            features_std = torch.std(self.features, dim=0).unsqueeze(0)
            self.features = torch.nan_to_num((self.features - features_mean) / features_std)
        ## normalize the data if needed
        if normalize:
            scaler = MinMaxScaler()
            self.features = scaler.fit_transform(self.features)
            self.features = torch.from_numpy(self.features).type(torch.float32)
        # put on GPU if available
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)

    def __getitem__(self, idx):
        return self.features[idx, :], self.labels[idx]
    
    def __len__(self):
        return self.features.shape[0]