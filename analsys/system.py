from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

import pandas as pd
from utils import *

class SkSystem:
	def __init__(self, params, fpath):
		print('SkSystem RUN', params)
		k, usePca, n, _ = params

		X_train, X_test, Y_train, _ = parse_training_data(fpath)
		self.X_train = X_train.to_numpy()
		self.X_test = X_test.to_numpy()
		self.Y_train = Y_train

		if usePca:
			self.calc_pca(n)
		self.guess(k)
		save_results(self.results, results_sklearn_fpath(fpath))

	def calc_pca(self, n): #U: Calculates the pca transform and transforms the training data to use it #TODO: niters?
		pca = PCA(n)
		pca.fit(self.X_train)
		self.X_train = pca.transform(self.X_train)
		self.X_test = pca.transform(self.X_test)

	def guess(self, k):
		clf = KNeighborsClassifier(k)
		clf.fit(self.X_train, self.Y_train)
		self.results = clf.predict(self.X_test)

