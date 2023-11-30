# ELEC 6410 (DSP) Project | Tanner Koza

This document serves to explain how to generate features in Python for classifying types of ADHD with an SVM. This markdown file can be viewed in its rendered version at this link: https://github.com/tannerkoza-auburn/dsp-project/tree/main.

## Usage

1. Open this project in a shell of your choice. Make sure you're in the `dsp-project/` directory.
2. Install packages.

```shell
pip install -r requirements.txt
```

3. Place file containing data to classify (eg. `data.mat`) in `data/` directory.

   > **NOTE**: If file name is not `data.mat`, open `scripts/compute_features.py` in an editor and change the `INPUT_FILE` parameter to your file name. **DO NOT** change anything else.

4. Run `compute_features.py` script. This will output a file called `features.mat` in the `data/` directory.

```shell
python3 scripts/compute_features.py
```

5. Load `data/features.mat` in MATLAB and process accordingly.

   > **ALSO:** The MATLAB script `scripts/test_features.m` was used to validate performance.
