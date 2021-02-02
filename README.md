# Non-Specific Syndromic Surveillance

## Installation
This project is written in Python 3.7 and R. The Python packages can be installed by using pip:
```sh
pip install numpy
pip install pandas
pip install tabulate
pip install scipy
pip install sklearn

# for DMSS (assoiciation rule learning)
pip install mlxtend

# for WSARE (Bayesian networks) 
pip install pyAgrum

# for anomaly detectors
pip install pyod
pip install tensorflow
pip install keras

# for sum-product networks
pip install spflow

# for specific syndromic surveillance methods
pip install rpy2
```

In order to execute the code for the specific syndromic surveillance methods (e.g. `algo.syndromic_surveillance` module), 
an installation of R is required. For R the following package need to be installed:
```sh
install.packages("surveillance")
```

## Data
The WSARE data can be downloaded 
[here](https://web.archive.org/web/20150920104314/http://www.autonlab.org/auton_static/datsets/wsare/wsare3data.zip)
and should be placed in the folder `_data.wsare`.

## Parameter Optimization (IDA Paper)
All possible combinations of the following parameters have been tuned for the anomaly detectors:

```
# Auto Encoder
    ["hidden_neurons", [[32, 16, 16, 32], [16, 8, 8, 16], [8, 4, 4, 8]]],
    ["epochs", [100, 500]],
    ["dropout_rate", [0.1, 0.2]],
    ["l2_regularizer", [0.1, 0.01, 0.005, 0.001]],
    ["random_state", [1]],
    ["output_activation", ['sigmoid', 'relu']],
    ["preprocessing", [True, False]]

# MO GAAL 
    ["k", [5, 10]],
    ["stop_epochs", [20, 200]],
    ["lr_d", [0.1]],
    ["lr_g", [0.001]],
    ["normalize_data", [False, True]
    AND    
    ["k", [5, 10]],
    ["stop_epochs", [20, 200]],
    ["lr_d", [0.01]],
    ["lr_g", [0.0001]],
    ["normalize_data", [False, True]
    AND
    ["k", [5, 10]],
    ["stop_epochs", [20, 200]],
    ["lr_d", [0.001]],
    ["lr_g", [0.00001]],
    ["normalize_data", [False, True]
    AND
    ["k", [5, 10]],
    ["stop_epochs", [20, 200]],
    ["lr_d", [0.0005]],
    ["lr_g", [0.000005]],
    ["normalize_data", [False, True]
    AND
    ["k", [5, 10]],
    ["stop_epochs", [20, 200]],
    ["lr_d", [0.0001]],
    ["lr_g", [0.000001]],
    ["normalize_data", [False, True]

# One-Class SVM #
    ["kernel", ["linear"]],
    ["nu", [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]],
    ["coef0", [0, 1]],
    ["normalize_data", [False, True]
    AND
    ["kernel", ["rbf"]],
    ["nu", [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]],
    ["degree", [2]],
    ["gamma", ['auto', 'scale']],
    ["coef0", [0]],
    ["normalize_data", [False, True]

# Local Outlier Factor 
    ["novelty", [True]],
    ["n_neighbors", [3, 5, 7, 10, 20, 40]],
    ["algorithm", ["auto"]],
    ["p", [1, 2, 3]],

# Gaussian Mixture 
    ["n_components", [1, 3, 5]],
    ["tol", [1e-2, 1e-3, 1e-4]],
    ["reg_covar", [1e-5, 1e-6, 1e-7]],
    ["max_iter", [100, 200]],
    ["random_state", [1]],

# Isolation Forest 
    ["n_estimators", [100, 300]],
    ["max_samples", [1.0, 0.75, 0.5]],
    ["max_features", [1.0, 0.75, 0.5]],
    ["bootstrap", [True, False]],
    ["random_state", [1]]
```


## Parameter Optimization (AIME Paper)
All possible combinations of the following parameters have been tuned for the anomaly detectors:

```
# Auto Encoder
    ["hidden_neurons", [[32, 16, 16, 32], [16, 8, 8, 16], [8, 4, 4, 8]]],
    ["epochs", [100, 500]],
    ["dropout_rate", [0.1, 0.2]],
    ["l2_regularizer", [0.1, 0.01, 0.005, 0.001]],
    ["random_state", [1]],
    ["output_activation", ['sigmoid', 'relu']],
    ["preprocessing", [True, False]]

# One-Class SVM #
    ["kernel", ["linear"]],
    ["nu", [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]],
    ["coef0", [0, 1]],
    ["normalize_data", [False, True]
    AND
    ["kernel", ["rbf"]],
    ["nu", [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]],
    ["degree", [2]],
    ["gamma", ['auto', 'scale']],
    ["coef0", [0]],
    ["normalize_data", [False, True]

# Gaussian Mixture 
    ["n_components", [1, 3, 5]],
    ["tol", [1e-2, 1e-3, 1e-4]],
    ["reg_covar", [1e-5, 1e-6, 1e-7]],
    ["max_iter", [100, 200]],
    ["random_state", [1]],

# SPN
    ["rdc", [0.5, 0.3, 0.1]]
    ["mis", [1.0, 0.7, 0.5, 0.3, 0.2, 0.1]]
    ["distributions", ["gaussian", "poisson", "nb"]]
    ["evidences", ["single", "single_double"]]
    ["product_combines", ["multiply", "fisher", "stouffer"]]
    ["sum_combines", ["weighted_average", "weighted_harmonic", "weighted_geometric"]]
```


## Citation

```
@InProceedings{mk:IDA-21,
  author    = {Moritz Kulessa and Loza Menc{\'i}a, Eneldo  and Johannes F{\"u}rnkranz},
  booktitle = {Proc. 19th Int.\ Symp.\ Intell. Data Anal. (IDA)},
  title     = {Revisiting Non-Specific Syndromic Surveillance},
  year      = {2021},
  note = {In press. \url{http://arxiv.org/abs/2101.12246}}
}
```