from utils import *
from system import SkSystem
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def metrics(true_results, results):
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

def print_pca(results, fpath):
	for k, (_, eigens, _) in results.items():
		fig = make_subplots(rows = ceil(len(eigens), 4), cols = 4)
		for i, eigen in enumerate(eigens):
			fig.add_trace(
				px.imshow(np.array(eigen).reshape(28, 28)).data[0],
				row = (i // 4) + 1,
				col = (i % 4) + 1,
			)
		layout = px.imshow(np.array(eigens[0]).reshape(28, 28), color_continuous_scale='gray').layout
		fig.layout.coloraxis = layout.coloraxis
		fig.write_image(img_fpath(f'{fpath}.{k}.eigens'))

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
	scores = metrics(true_results, results)
	print_pca(results, fpath)
	print(scores)
