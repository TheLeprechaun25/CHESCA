# CHESCA
Code for the paper "Winning the 2023 CityLearn Challenge: a Community-based Hierarchical Energy Systems Coordination Algorithm"


## Requirements
- Python
- citylearn (pip install CityLearn==2.1b12 for 2023 challenge environment)
- numpy
- xgboost

## Local evaluation
Run the following to locally evaluate the control algorithm
```bash
python local_evaluation.py
```

In order to evaluate using a different dataset, change the SCHEMA path to the path of the schema you want to test. 

See data/schemas/ for available schemas.
