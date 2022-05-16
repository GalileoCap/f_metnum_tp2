import tp2
import sklearn.decomposition
import sklearn.neighbors

import time
import metrics
from utils import *

class System:
	def __init__(self, whose):
		self.whose = whose
	
	def fit(self, params, X_train, Y_train, *, skipPCA = False):
		#TODO: Check correct size
		self.params = params

		self.usePCA = (params['n'] > 0) or skipPCA #TODO: Repetitive redundancy
		if self.usePCA and not skipPCA:
			start = time.process_time()

			self.pca = tp2.PCA(params['n'], params['maxIters']) if self.whose == 'mine' else sklearn.decomposition.PCA(params['n'], iterated_power = params['maxIters'])
			self.pca.fit(X_train)

			self.pcaTime = time.process_time() - start
	
		if self.usePCA or skipPCA: X_train = self.pca.transform(X_train)
		self.knn = tp2.KNN(params['k']) if self.whose == 'mine' else sklearn.neighbors.KNeighborsClassifier(params['k'])
		self.knn.fit(X_train, Y_train)

	def predict(self, X_test):
		#TODO: Check correct size
		start = time.process_time()

		if self.usePCA:
			X_test = self.pca.transform(X_test)
		self.predicted = self.knn.predict(X_test)

		self.predictTime = time.process_time() - start

	def scores(self, Y_test):
		#TODO: Check correct size
		scores = metrics.scores(Y_test, self.predicted)
		scores.update(self.params)
		scores.update({
			'whose': self.whose,
			'pcaTime': self.pcaTime if self.usePCA else np.nan,
			'predictTime': self.predictTime,
		})
		return scores

class Systems:
	def __init__(self, df):
		self.systems = [System(whose) for whose in ['mine', 'sklearn']]
		self.df = df
		self.params = dict() #A: Dflt
	
	def run(self, params): #U: Runs the system with some specific params 
		#S: Checks which params changed
		newDataSz = self.params.get('dataSz') != params['dataSz']
		newFrac = self.params.get('frac') != params['frac']
		newMaxIters = self.params.get('maxIters') != params['maxIters']
		newN = self.params.get('n') != params['n']
		newK = self.params.get('k') != params['k']
		self.params = params #A: Save the new params
	
		if newDataSz or newFrac: self.splitData() #A: If the size changed, we need to re-split the data
		self.fit(
			skipPCA = (params['n'] > 0) and (not (newDataSz or newFrac or newMaxIters or newN)) #A: We're using PCA but we don't need to train it 
		)
		self.predict()
		return self.scores()

	def splitData(self): #U: Gets a subset of df and splits it in training and testing data
		df = self.df.drop(np.random.choice( #A: Choose dataSz rows at random
			self.df.index,
			len(self.df) - self.params['dataSz'],
			replace = False
		))
		train = df.sample(frac = self.params['frac']) #A: Split train and test 
		test = df.drop(train.index)

		self.X_train = train.drop('label', axis = 1).to_numpy().astype(float) #A: Split into vectors and labels
		self.X_test = test.drop('label', axis = 1).to_numpy().astype(float)
		self.Y_train = train['label'].to_numpy().astype(np.uint64)
		self.Y_test = test['label'].to_numpy().astype(np.uint64)

	def fit(self, *, skipPCA = False):
		for sys in self.systems:
			sys.fit(self.params, self.X_train, self.Y_train, skipPCA = skipPCA)

	def predict(self):
		for sys in self.systems:
			sys.predict(self.X_test)

	def scores(self):
		return [sys.scores(self.Y_test) for sys in self.systems]
