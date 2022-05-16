from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from utils import *

def scores(Y_test, predicted):
	labels, average = range(0, 10), 'macro' #TODO: Labels as a parameter
	return {
		'confusionMatrix': confusion_matrix(Y_test, predicted, labels = labels),
		'accuracy': accuracy_score(Y_test, predicted), #TODO: Balanced?
		'precision': precision_score(Y_test, predicted, labels = labels, average = average), 
		'recall': recall_score(Y_test, predicted, labels = labels, average = average),
		'f1': f1_score(Y_test, predicted, labels = labels, average = average),
		'cohenKappa': cohen_kappa_score(Y_test, predicted, labels = labels),
	}
