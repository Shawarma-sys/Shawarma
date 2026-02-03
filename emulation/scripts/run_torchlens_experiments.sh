#!/bin/bash

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

# Run the experiment 10 times
for i in {1..10}
do
    echo "Starting experiment $i of 10..."
    
    # Start control plane experiment in background
    echo "Starting control plane experiment..."
    python3 experiments/control_plane_torchlens_experiment.py &
    control_pid=$!

    # Wait for control plane to initialize (5 seconds should be enough)
    echo "Waiting for control plane to initialize..."
    sleep 5

    # Start data plane experiment
    echo "Starting data plane experiment..."
    python3 experiments/data_plane_torchlens_experiment.py

    # After data plane finishes, kill the control plane process
    echo "Data plane experiment completed, stopping control plane..."
    kill $control_pid

    echo "Experiment $i completed!"
    
    # Add a small delay between experiments
    if [ $i -lt 10 ]; then
        echo "Waiting 3 seconds before starting next experiment..."
        sleep 3
    fi
done

echo "All 10 experiments completed!"
