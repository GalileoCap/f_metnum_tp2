from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from utils import *

def scores(true_results, results):
	labels, average = range(1, 10), 'macro' #TODO: Labels as a parameter
	(_, _, v) = results
	return {
		'confusionMatrix': confusion_matrix(true_results, v, labels = labels),
		'accuracy': accuracy_score(true_results, v), #TODO: Balanced?
		'precision': precision_score(true_results, v, labels = labels, average = average), 
		'recall': recall_score(true_results, v, labels = labels, average = average),
		'f1': f1_score(true_results, v, labels = labels, average = average),
		'cohenKappa': cohen_kappa_score(true_results, v, labels = labels),
	}

def time(results, usePCA):
	((pcaTime, times), _, _) = results
	return {
		'pcaTime': pcaTime if usePCA else np.nan,
		'guessTime': np.sum(times) + (pcaTime if usePCA else 0), #A: Fix wrongfully split
	}
