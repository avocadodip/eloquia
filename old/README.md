# Eloquia: a machine learning tool to identify disfluencies in speech

## Installation
1. Clone Eloquia github repo
   ```console
   git clone git@github.com:adikondepudi/eloquia.git
   ```
2. Install miniconda: follow instructions
    [here](https://docs.conda.io/en/latest/miniconda.html)
3. Create conda environment
    ```console
    conda create -n eloquia python=3.9
    ```
4. Activate conda environment
    ```console
    conda activate eloquia
    ```
5. Install package and dependencies
    ```console
    <cd /path/to/eloquia/repo/dir>
    pip install -e .
    ```

## Directory organization
- `src/`: 
    - `models/`: models for training and evaluation
    - `train/`: training and evaluation scripts
- `utils/`: utility scripts and helper functions
- `data/`: raw audio files and labels
- `README.md`
- `LICENSE`

