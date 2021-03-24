# mimic-lstm

<a href="https://zenodo.org/badge/latestdoi/155128190"><img src="https://zenodo.org/badge/155128190.svg" alt="DOI"></a>

This is a complete preprocessing, model training, and figure generation repo for an adapted version of "An attention based deep learning model of clinical events in the intensive care unit". It allows the use of MIMIC-III and MIMIC-IV and produces PyTorch models.

### Getting Started

To begin, clone the mimic-lstm repository. Within the repository, create a folder called, "data/mimic_X_database" where X is either 3 or 4 depending on your MIMIC version. Then put all MIMIC CSVs into the folder.

The last thing that needs to be done is setting the variables in train.py to your preferences. The program will create the rest of the folders itself.

Models and figures are generated in the test.ipynb notebook. Simply adjusting the target to 'MI', 'Sepsis', or 'Vancomycin' will generate the figures panels and images required for each part of the figure.

### Prerequisites
Refer to mimic-lstm-env.yml

### License
This project is licensed under the MIT License - see the LICENSE.md file for details

### Acknowledgments
We thank Deepak A. Kaji for laying the groundwork of this implementation in "An attention based deep learning model of clinical events in the intensive care unit"
