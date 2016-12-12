# For saving / loading processed datasets
import pickle

# For feeding the data to the Keras model
import numpy as np

# For reading captured .jpg files into numpy arrays
from scipy import misc

# For checking files and paths
import os

# For reading csv output
import pandas as pd

def makeInputPickle():
    '''
    Depending on the size of your input, this could exhaust memory :(
    '''
    
    im_array = []

    for track in ['T1', 'T2']:
        for run in ['R01', 'R02', 'R03', 'R04', 'R05', 'R06', 'R07', 
                    'R08', 'R09', 'R10', 'R11', 'R12', 'R13', 'R14']:
            if not os.path.isdir(track + '/' + run):
                continue
            else:
                fp = track + '/' + run + '/' + 'IMG'
                imgs = os.listdir(fp)
                for f in imgs:
                    if 'center' in f:
                        im_array.append(misc.imread(fp + '/' + f))
    im_array = np.array(im_array)
    
    pf = 'pickles/X.p'
    try:
        pd = open(pf, 'wb')
        pickle.dump(np.asarray(im_array), pd, pickle.HIGHEST_PROTOCOL)
        pd.close()
    except Exception as e:
        print('Unable to save data to', pf, ':', e)
        raise

def makeLabelsPickle():
    '''
    '''
    
    
