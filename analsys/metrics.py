from utils import *
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

def scores(true_results, results):
	scores = dict()
	for k, (_, _, v) in results.items():
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

