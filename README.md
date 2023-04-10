# MIPT ML CV HW2

==============================

Face recognition system as a part of ML MIPT homework

## Getting started

```Bash
cd FaceRec/
pip install -e .
source ./set_envars.sh
source ./start_app.sh
```

## Project Organization

------------

```Code
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained models
    │
    ├── notebooks          <- Jupyter notebooks
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   │
    │   │   ├── preprocessing.py 
    │   │   └── dataset.py
    │   │
    │   ├── models         <- Model architectures
    │   │   │                 
    │   │   ├── swinnet.py
    │   │   ├── swinnetv2.py
    │   │   ├── efficientnetb0.py
    │   │   └── efficientnetb3.py
    │   │
    │   └── pipeline       <- Scripts to create exploratory and results oriented visualizations
    │   │   │                 
    │   │   ├── save_load_model.py
    │   │   └── train_model.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
```
