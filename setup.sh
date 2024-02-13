#!/bin/bash

# Step 1: Create a virtual environment named 'env'
python3 -m venv venv

# Step 2: Activate the environment
source venv/bin/activate

# Step 3: Install packages from requirements.txt
pip install -r requirements.txt

pip install "unsloth[cu121_torch220] @ git+https://github.com/unslothai/unsloth.git"
