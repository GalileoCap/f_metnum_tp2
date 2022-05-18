from system import System
import numpy as np

def knn(): 
	X = np.array([
		[0, 0., 0.],
		[1, 0., 1.],
		[2, 1., 0.],
		[3, 1., 1.],
	])
	X_train, Y_train = X[:, 1:].copy(), X[:, 0].copy().astype(np.uint64)
	params = {'maxIters': pow(10, 6), 'n': -1, 'k': 1}
	sys = System('mine')
	sys.fit(params, X_train, Y_train)

	X_test, Y_test = X_train.copy(), Y_train.copy()
	sys.predict(X_test)
	assert (np.array_equal(sys.predicted, Y_test)), f'Basic guessing {sys.predicted}'

	np.random.shuffle(X)
	X_test, Y_test = X[:, 1:].copy(), X[:, 0].copy().astype(np.uint64)
	sys.predict(X_test)
	if not np.array_equal(sys.predicted, Y_test):
		print('Tricked by shuffling', X_train, X_test, sys.predicted, sep = '\n')

def pca(): 
	X = np.array([
		[0, 0., 0.],
		[1, 0., 1.],
		[2, 1., 0.],
		[3, 1., 1.],
	])
	X_train, Y_train = X[:, 1:], X[:, 0].astype(np.uint64)
	X_test, Y_test = X_train, Y_train
	sys = System('mine')

	params = {'maxIters': pow(10, 6), 'n': 1, 'k': 1}
	sys.fit(params, X_train, Y_train)
	sys.predict(X_test)
	assert (np.array_equal(sys.predicted, Y_test)), 'n = 1'

	params = {'maxIters': pow(10, 6), 'n': 2, 'k': 1}
	sys.fit(params, X_train, Y_train)
	sys.predict(X_test)
	assert (np.array_equal(sys.predicted, Y_test)), 'n = 2'

if __name__ == '__main__':
	knn()
