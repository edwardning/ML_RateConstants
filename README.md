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

*
If one wants to train one's own model to generate reaction fingerprints, please refer to:

*ML_RateConstants/rxnfp_schwaller/fine_tune.py*  

*  
More details on the reaction fingerprints generation, please go to:  

*https://rxn4chemistry.github.io/rxnfp//index.html*
