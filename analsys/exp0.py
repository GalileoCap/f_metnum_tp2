import metrics
import plot
from system import SkSystem
import pandas as pd
from utils import *

def exp0_fpath(fpath):
	return f'{fpath}.exp0'

def exp0_run_instance(params, fpath):
	run(params, fpath) #A: Run c++ code
	SkSystem(params, fpath) #A: Run sklearn code
	results = {
		'mine': parse_results(results_mine_fpath(fpath)),
		'sklearn': parse_results(results_sklearn_fpath(fpath)),
	}

	(k, _, n, _) = params
	_, _, _, true_results = parse_training_data(fpath)
	scores = metrics.scores(true_results, results)
	for s in scores:
		s.update({'k': k, 'n': n})
	return scores 

def exp0_run(range_k, range_n, fpath):
	print(f'exp0 RUN')
	scores = []
	for k in range_k:
		for n in range_n:
			params = (k, n > 0, n, 1000000) #TODO: Change niters
			scores += exp0_run_instance(params, fpath)
						
	df = pd.DataFrame(scores)
	df.to_csv(pandas_fpath(exp0_fpath(fpath)), index = False) 
	return df

def exp0(range_k, range_n, fpath, replace): #U: Runs the program with different values for k and a #TODO: Rename
	print(f'exp0 START range_k {range_k}, range_n {range_n}, fpath {fpath}, replace {replace}')
	if not replace:
		try:
			df = pd.read_csv(pandas_fpath(exp0_fpath(fpath)))
		except FileNotFoundError: #A: If the file doesn't exist we need to replace anyways
			print('exp0 ERROR file not found, replacing anyways')
			df = exp0_run(range_k, range_n, fpath)
		else:
			#TODO: confusion_matrix not read correctly
			#TODO: Check if read correct file
			pass
	else:
		df = exp0_run(range_k, range_n, fpath)
	plot.exp0(df, fpath)

if __name__ == '__main__':
	fpath = '../data/kaggle/small'
	split_data(0.4, fpath)
	exp0(range(10, 50 + 1, 10), range(0, 10 + 1), fpath, True)
