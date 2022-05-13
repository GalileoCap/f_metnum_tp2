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

def exp_path(name, exp): #U: Gets the path to a dataset's experiments
	return make_data_path(f'{name}/experiments/{exp}')

def df_fpath(path, df): #U: Gets the path to a df
	return f'{make_data_path(path)}/{df}.csv.gz'

def full_fpath(name): #U: Gets the path to the original dataset
	return f'{get_data_path()}/{name}.csv.gz'

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

def split_data(name, path, frac = 0.4, subset = None): #U: Splits an original dataset (name) and saves the result in path, if subset, only keeps #subset rows from the original df
	df = pd.read_csv(full_fpath(name))
	if not subset is None: #TODO: Read subset random rows directly from the csv
		df = df.drop(np.random.choice(df.index, len(df) - subset, replace = False))
	train = df.sample(frac = frac) #A: Split train and test 
	test = df.drop(train.index)

	train.to_csv(train_fpath(path), index = False) #A: Save them
	test.to_csv(test_fpath(path), index = False)

def parse_training_data(path):
	train = pd.read_csv(train_fpath(path))
	test = pd.read_csv(test_fpath(path))

	X_train = train.drop('label', axis = 1) #A: Split further for sklearn 
	X_test = test.drop('label', axis = 1)
	Y_train = train['label'].to_numpy()
	Y_test = test['label'].to_numpy()

	return X_train, X_test, Y_train, Y_test

def parse_results(path, whose):
	with open(results_fpath(path, whose), 'r') as fin:
		lines = fin.read().splitlines()
		print(lines[0])
	times = [int(x) for x in lines[0].split()]
	eigens = [[float(x) for x in line.split()] for line in lines[1:-1]]
	results = [int(x) for x in lines[-1].split()]
	return (
		(times[0], times[1:]), #A: Split PCA time from guesses
		eigens,
		np.array(results)
	)

#TODO: Rename variable name
