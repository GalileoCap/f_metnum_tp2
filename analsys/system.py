import subprocess

from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

import metrics
from utils import *

class MySystem:
	def __init__(self, params, path):
		print(f'MySystem RUN params {params}, path {path}')
		k, usePCA, n, niters = params
		args = [
			train_fpath(path),
			test_fpath(path),
			results_fpath(path, 'mine'),
			f'-k {k}',
		] + ([f'-n {n}', f'-i {niters}'] if usePCA else [])
		subprocess.run(f'../tp2 {" ".join(args)}', shell = True, capture_output = True) #TODO: Check if error
		self.results = parse_results(path, 'mine') 

		X_train, X_test, _, true_results = parse_training_data(path)
		self.metrics = metrics.scores(true_results, self.results)
		self.metrics.update(metrics.time(self.results, usePCA))
		self.metrics.update({
			'whose': 'mine',
			'k': k, 'n': n, 'niters': niters,
			'trainSz': len(X_train), 'testSz': len(X_test),
		})

class SkSystem:
	def __init__(self, params, path):
		print(f'SkSystem RUN params {params}, path {path}')
		k, self.usePCA, n, niters = params
		self.path = path

		X_train, X_test, Y_train, true_results = parse_training_data(path)
		self.X_train = X_train.to_numpy()
		self.X_test = X_test.to_numpy()
		self.Y_train = Y_train

		if self.usePCA:
			self.calc_pca(n)
		self.guess(k)
		self.save_results()
		self.results = parse_results(path, 'sklearn') #TODO: We already have results, but I want to make sure they're the correct type

		self.metrics = metrics.scores(true_results, self.results)
		self.metrics.update(metrics.time(self.results, self.usePCA))
		self.metrics.update({
			'whose': 'sklearn',
			'k': k, 'n': n, 'niters': niters,
			'trainSz': len(X_train), 'testSz': len(X_test),
		})

	def calc_pca(self, n): #U: Calculates the pca transform and transforms the training data to use it #TODO: niters?
		self.pca = PCA(n)
		self.pca.fit(self.X_train)
		self.X_train = self.pca.transform(self.X_train)
		self.X_test = self.pca.transform(self.X_test)

	def guess(self, k):
		clf = KNeighborsClassifier(k)
		clf.fit(self.X_train, self.Y_train)
		self.results = clf.predict(self.X_test)

	def save_results(self):
		with open(results_fpath(self.path, 'sklearn'), 'w') as fout:
			fout.write('0 0\n') #NOTE: Ignore this line
			if self.usePCA:
				fout.write('\n'.join([
					' '.join([str(x) for x in eigenvector])
					for eigenvector in self.pca.components_
				])) #A: n lines of PCA's eigenvectors
			fout.write('\n') 
			fout.write(' '.join([str(x) for x in self.results])) #A: Line of results

