import numpy as np
import pandas as pd

import os
import itertools as itt

def get_data_path():
	return '../data'

def make_data_path(path): #U: Makes sure a path exists
	path = f'{get_data_path()}/{path}'
	os.makedirs(path, exist_ok = True)
	return path

def exp_path(dataset, exp, name): #U: Gets the path to a dataset's experiments
	return make_data_path(f'{dataset}/experiments/{exp}/{name}')

def df_fpath(path, df): #U: Gets the path to a df
	return f'{make_data_path(path)}/{df}.csv.gz'

def full_fpath(dataset): #U: Gets the path to the original dataset
	return f'{get_data_path()}/{dataset}.csv.gz'

def train_fpath(path): #U: Gets the path to a training dataset
	return f'{make_data_path(path)}/train.csv'

def test_fpath(path): #U: Gets the path to a test dataset
	return f'{make_data_path(path)}/test.csv'

def results_fpath(path, whose): #U: Gets the path to someone's results
	return f'{make_data_path(path)}/{whose}.results'

def img_fpath(fpath): #U: Adds the correct image extension
	return f'{fpath}.png'

def ceil(x):
	return int(x) + (int(x) < x)
