# For saving / loading processed datasets
import pickle

# For feeding the data to the Keras model
import numpy as np

# For reading captured .jpg files into numpy arrays
from scipy import misc

# For checking files and paths
import os

im_array = []

for track in ['T1', 'T2']:
    for run in ['R01', 'R02', 'R03', 'R04', 'R05', 'R06', 'R07', 
                'R08', 'R09', 'R10', 'R11', 'R12', 'R13', 'R014']:
    
