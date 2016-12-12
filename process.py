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

def logsToDataframe():
    '''
    Depending on the size of your input, this could exhaust memory :(
    '''
    
    merged_df = pd.DataFrame()
    
    for track in ['T1', 'T2']:
        for run in ['R01', 'R02', 'R03', 'R04', 'R05', 'R06', 'R07', 
                    'R08', 'R09', 'R10', 'R11', 'R12', 'R13', 'R14']:
            if not os.path.isdir(track + '/' + run):
                continue
            else:
                fp = track + '/' + run + '/' + 'driving_log.csv'
                df = pd.read_csv(fp, usecols=[0,3], names=['X_file','y_angle'])
                merged_df = merged_df.append(df, ignore_index=True)
    
    return merged_df
    
def makePickle(filename, dataframe, column)    
    pf = 'pickles/' + filename
    dataframe.to_pickle(pf)

def makeLabelsPickle():
    '''
    '''
    
    
