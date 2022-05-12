from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

import pandas as pd
from utils import *

class SkSystem:
	def __init__(self, params, fpath):
		print('SkSystem RUN', params)
		k, self.usePCA, n, _ = params

		X_train, X_test, Y_train, _ = parse_training_data(fpath)
		self.X_train = X_train.to_numpy()
		self.X_test = X_test.to_numpy()
		self.Y_train = Y_train

		if self.usePCA:
			self.calc_pca(n)
		self.guess(k)
		self.save_results(results_sklearn_fpath(fpath))

	def calc_pca(self, n): #U: Calculates the pca transform and transforms the training data to use it #TODO: niters?
		self.pca = PCA(n)
		self.pca.fit(self.X_train)
		self.X_train = self.pca.transform(self.X_train)
		self.X_test = self.pca.transform(self.X_test)

	def guess(self, k):
		clf = KNeighborsClassifier(k)
		clf.fit(self.X_train, self.Y_train)
		self.results = clf.predict(self.X_test)

	def save_results(self, fpath):
		with open(fpath, 'w') as fout:
			fout.write('0 0\n') #A: Empty line instead of times
			if self.usePCA:
				fout.write('\n'.join([
					' '.join([str(x) for x in eigenvector])
					for eigenvector in self.pca.components_
				])) #A: n lines of PCA's eigenvectors
			fout.write('\n') 
			fout.write(' '.join([str(x) for x in self.results])) #A: Line of results

