from utils import *
from system import SkSystem
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

def metrics(true_results, results):
	scores = dict()
	for k, v in results.items():
		labels, average = range(1, 10), 'macro' #TODO: Labels as a parameter
		scores[k] = {
			'confusion_matrix': confusion_matrix(true_results, v, labels = labels),
			'accuracy': accuracy_score(true_results, v), #TODO: Balanced?
			'precision': precision_score(true_results, v, labels = labels, average = average), 
			'recall': recall_score(true_results, v, labels = labels, average = average),
			'f1': f1_score(true_results, v, labels = labels, average = average),
			'cohen_kappa': cohen_kappa_score(true_results, v, labels = labels),
		}
	return scores

if __name__ == '__main__':
	params, pct, replace = (5, False, None, None), 0.3, True
	fpath = '../data/kaggle/small'

	if replace:
		split_data(pct, fpath)
		run(params, fpath) #A: Run c++ code
		SkSystem(params, fpath) #A: Run sklearn code

	true_results = parse_training_data(fpath)[3]
	results = {
		'mine': parse_results(results_mine_fpath(fpath))[1],
		'sklearn': parse_results(results_sklearn_fpath(fpath))[1],
	}
	scores = metrics(true_results, results)
	print(scores)
