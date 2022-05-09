import subprocess
import numpy as np
import pandas as pd

def csv_fpath(fpath):
	return f'{fpath}.csv'

def train_fpath(fpath):
	return f'{fpath}.train'

def test_fpath(fpath):
	return f'{fpath}.test'

def result_fpath(fpath):
	return f'{fpath}.result'

def split_data(fpath, pct = 0.4):
	df = pd.read_csv(csv_fpath(fpath))
	train = df.sample(frac = pct) #A: Split train and test 
	test = df.drop(train.index)

	X_train = train.drop('label', axis = 1) #A: Split further for sklearn 
	X_test = test.drop('label', axis = 1)
	Y_train = train['label'].to_numpy()
	Y_test = test['label'].to_numpy()

	train.to_csv(csv_fpath(train_fpath(fpath)), index = False) #A: Save training data
	X_test.to_csv(csv_fpath(test_fpath(fpath)), index = False) #A: Save test data without labels

	return X_train, X_test, Y_train, Y_test

def parse_results(fpath):
	with open(fpath, 'r') as fin:
		times, results = [[int(x) for x in line.split()] for line in fin.read().split('\n')]
	return (
		(times[0], times[1:]), #A: Split PCA time from guesses
		results
	)


def run(fpath, k = 5, usePca = False, n = None, niters = None):
	args = [
		f'./{csv_fpath(train_fpath(fpath))}',
		f'./{csv_fpath(test_fpath(fpath))}',
		f'./{result_fpath(fpath)}',
		str(k),
	] + ([str(n), str(niters)] if usePca else [])
	subprocess.run(f'../tp2 {" ".join(args)}', shell = True)

	return parse_results(result_fpath(fpath))
