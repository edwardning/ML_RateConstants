# ML_RateConstants
> Using machine learning to predict rate constants for various reactions in combustion models

## How to use

It is recommended to create a new `conda` environment to run this project

```console
conda create -n ml_k python=3.7
conda activate ml_k
pip install -r requirements.txt
```

Before training the model, prepare the dataset using the following script:

*ML_RateConstants/rxnfp_schwaller/generate_rxnfp.py*
