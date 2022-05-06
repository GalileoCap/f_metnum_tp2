from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
import pandas as pd

class SkSystem:
	def __init__(self, data, k, usePca, n):
		X_train, X_test, Y_train, _ = data
		self.X_train = X_train.to_numpy()
		self.X_test = X_test.to_numpy()
		self.Y_train = Y_train

		if usePca:
			self.calc_pca(n)
		self.guess(k)

	def calc_pca(self, n): #U: Calculates the pca transform and transforms the training data to use it #TODO: niters?
		pca = PCA(n)
		pca.fit(self.X_train)
		self.X_train = pca.transform(self.X_train)
		self.X_test = pca.transform(self.X_test)

	def guess(self, k):
		clf = KNeighborsClassifier(k)
		clf.fit(self.X_train, self.Y_train)
		self.results = clf.predict(self.X_test)

