## Installation
1) Create a `conda` environment.

2) Install various dependencies: `pip install ipykernel numpy pandas matplotlib librosa soundfile pyroomacoustics tqdm`.

3) Install PyTorch (check [the official docs](https://pytorch.org/get-started/locally)): `pip install torch torchaudio torchvision torchcodec`.

## Before Running
Make sure you have at least __225 GB__ of free memory space.  
- The EARS dataset (raw + preprocessed) takes up approx. 75 GB.
- So does the WHAM! dataset (raw + preprocessed).
- The resulting MIX-EARS-WHAM database takes up approx. 175 GB.

## How to Use
1) Download the EARS dataset using [github.com/facebookresearch/ears_dataset](https://github.com/facebookresearch/ears_dataset).

2) Adjust root paths in `src/1-preprocessing/preprocess_ears.ipynb` and run.  
The output directory should contain subsets `trn`, `tst` and `val`, each containing thousands of clean speech samples.

3) Download the WHAM! dataset from [http://wham.whisper.ai/](http://wham.whisper.ai/).

4) Adjust root paths in `src/1-preprocessing/preprocess_wham.ipynb` and run.  
The output directory should contain subsets `trn`, `tst` and `val`, each containing thousands of noise samples.

5) Adjust root paths in `src/2-generation/generate.py` and run.  
The output directory should contain subsets `trn`, `tst` and `val`, each containing thousands of sample folders `00001`, `00002`, etc.
