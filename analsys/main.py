from system import SkSystem
import plot
import metrics
from utils import *

if __name__ == '__main__':
	params, pct, replace = (50, True, 12, 1000), 0.4, False
	fpath = '../data/kaggle/small'

	if replace:
		split_data(pct, fpath)
		run(params, fpath) #A: Run c++ code
		SkSystem(params, fpath) #A: Run sklearn code

	_, _, _, true_results = parse_training_data(fpath)
	results = {
		'mine': parse_results(results_mine_fpath(fpath)),
		'sklearn': parse_results(results_sklearn_fpath(fpath)),
	}
	scores = metrics.scores(true_results, results)
	plot.eigens(results, fpath)

	print(scores)
