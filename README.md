# Self-Driving-Car-Simulator
This code utilises the data collected using [car simulator](https://github.com/udacity/self-driving-car-sim) to train a CNN model to learn to auto drive a car on the track

---

## Dependencies

You can install all dependencies by running one of the following commands

You need a [anaconda](https://www.continuum.io/downloads) or [miniconda](https://conda.io/miniconda.html) to use the environment setting.

```python

conda env create -f environments.yml

```
---
## Usage

### Train the model

Visit [This Kaggle Notebook](https://www.kaggle.com/code/prathamsaraf1389/car-simulator)
and modify the code accordingly


### Test the model

* Run the car simulator in autonomous mode 

* Test one of the models by running the the script:

```python

python drive.py <path to model_file>

#Example
python drive.py models/model_50000

```

---

## File structure of the project
```
.
├── car-simulator.ipynb
├── car_simulator.py
├── Data
│   └── driving_log.csv
├── drive.py
├── environments.yml
├── models
│   ├── model_0
|   .
|   .
│   └── model_195000
└── README.md
```

## Credits

Credits to the paper ['End to End Learning for Self-Driving Cars'](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/) from Bojarski et al.