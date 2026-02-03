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

import torch
from torch import nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    
    
# the MLP model used as the labeler of the CIC-IDS2017/2018 dataset
# input features are consistent with the pForest paper
class MLP1Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )
        self.linear_relu_stack.apply(init_weights)

    def forward(self, x):
        sigmoid_score = self.linear_relu_stack(x)
        return sigmoid_score

    def run_inference(self, input_features):
        score = self.forward(input_features)
        label = torch.argmax(score, dim=1)
        return score, label
    

# the MLP model used in the data plane of the CIC-IDS2017/2018 dataset
# input features are consistent with the pForest paper
class MLP1Teacher_multi(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )
        self.linear_relu_stack.apply(init_weights)

    def forward(self, x):
        sigmoid_score = self.linear_relu_stack(x)
        return sigmoid_score

    def run_inference(self, input_features):
        score = self.forward(input_features)
        label = torch.argmax(score, dim=1)
        return score, label


# the MLP model used in the data plane of the CIC-IDS2017/2018 dataset
# input features are consistent with the pForest paper
class MLP1(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )
        self.linear_relu_stack.apply(init_weights)

    def forward(self, x):
        sigmoid_score = self.linear_relu_stack(x)
        return sigmoid_score

    def result_after_first_layer(self, x):
        return self.linear_relu_stack[1](self.linear_relu_stack[0](x))

    def result_after_second_layer(self, x):
        y = self.result_after_first_layer(x)
        return self.linear_relu_stack[3](self.linear_relu_stack[2](y))

    def result_after_third_layer(self, x):
        z = self.result_after_second_layer(x)
        return self.linear_relu_stack[4](z)

    def run_inference(self, input_features):
        score = self.forward(input_features)
        label = torch.argmax(score, dim=1)
        return score, label