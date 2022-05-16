import tp2
import sklearn.decomposition
import sklearn.neighbors

# import metrics
# from utils import *

class System:
	def __init__(self, whose):
		self.whose = whose
	
	def fit(self, params, X_train, Y_train):
		#TODO: Check correct size
		(k, n, maxIters), self.params = params, params
		self.trainSz = len(X_train)

		self.usePCA = n > 0
		if self.usePCA:
			self.pca = tp2.PCA(n, maxIters) if self.whose == 'mine' else sklearn.decomposition.PCA(n) #TODO: sklearn maxIters
			self.pca.fit(X_train)
			X_train = self.pca.transform(X_train)

		self.knn = tp2.KNN(k) if self.whose == 'mine' else sklearn.neighbors.KNeighborsClassifier(k)
		self.knn.fit(X_train, Y_train)

	def predict(self, X_test):
		#TODO: Check correct size
		self.testSz = len(X_test)

		if self.usePCA:
			X_test = self.pca.transform(X_test)
		self.predicted = self.knn.predict(X_test)

	def scores(self, Y_test):
		return self.predicted
		#TODO: Check correct size
		scores = metrics.scores(Y_test, self.predicted)
		# scores.update(metrics.time(self.results, self.usePCA)) #TODO
		(k, n, maxIters) = self.params = params
		scores.update({
			'whose': self.whose,
			'k': k, 'n': n, 'maxIters': maxIters,
			'trainSz': trainSz, 'testSz': testSz,
		})
		return scores

class Systems:
	def __init__(self):
		self.systems = [System(whose) for whose in ['mine', 'sklearn']]
	
	def fit(self, params, X_train, Y_train):
		for sys in self.systems:
			sys.fit(params, X_train, Y_train)

	def predict(self, X_test):
		for sys in self.systems:
			sys.predict(X_test)

	def scores(self, Y_test):
		res = []
		for sys in self.systems:
			res.append(sys.scores(Y_test))
		return res
