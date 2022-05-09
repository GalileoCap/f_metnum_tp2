import subprocess
import numpy as np
import pandas as pd

def csv_fpath(fpath):
	return f'{fpath}.csv'

def train_fpath(fpath):
	return f'{fpath}.train'

def test_fpath(fpath):
	return f'{fpath}.test'

def results_true_fpath(fpath):
	return f'{fpath}.true.results'

def results_mine_fpath(fpath):
	return f'{fpath}.mine.results'

def results_sklearn_fpath(fpath):
	return f'{fpath}.sklearn.results'

def split_data(pct, fpath):
	df = pd.read_csv(csv_fpath(fpath))
	train = df.sample(frac = pct) #A: Split train and test
	test = df.drop(train.index)

	train.to_csv(csv_fpath(train_fpath(fpath)), index = False) #A: Save them
	test.to_csv(csv_fpath(test_fpath(fpath)), index = False)

def parse_training_data(fpath):
	train = pd.read_csv(csv_fpath(train_fpath(fpath)))
	test = pd.read_csv(csv_fpath(test_fpath(fpath)))

	X_train = train.drop('label', axis = 1) #A: Split further for sklearn 
	X_test = test.drop('label', axis = 1)
	Y_train = train['label'].to_numpy()
	Y_test = test['label'].to_numpy()

	return X_train, X_test, Y_train, Y_test

def save_results(results, fpath):
	with open(fpath, 'w') as fout:
		fout.write('0 0\n') #A: Empty line instead of times
		fout.write(' '.join([str(x) for x in results])) #A: Line of results

def parse_results(fpath):
	with open(fpath, 'r') as fin:
		times, results = [[int(x) for x in line.split()] for line in fin.read().split('\n')]
	return (
		(times[0], times[1:]), #A: Split PCA time from guesses
		np.array(results)
	)

def run(params, fpath):
	k, usePca, n, niters = params
	args = [
		f'./{csv_fpath(train_fpath(fpath))}',
		f'./{csv_fpath(test_fpath(fpath))}',
		f'./{results_mine_fpath(fpath)}',
		str(k),
	] + ([str(n), str(niters)] if usePca else [])
	subprocess.run(f'../tp2 {" ".join(args)}', shell = True)

	return parse_results(results_mine_fpath(fpath))
