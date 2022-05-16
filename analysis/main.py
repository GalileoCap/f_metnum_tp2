from system import System

import numpy as np

if __name__ == '__main__':
	X = np.array([
		[0., 0., 0],
		[0., 1., 1],
		[1., 0., 2],
		[1., 1., 3]
	])
	X_train, Y_train = X[:, :-1], X[:, -1]
	X_test, Y_test = X_train, Y_train

	params = {
		'dataSz': len(X_train) + len(X_test), 'frac': 0.5,
		'maxIters': pow(10, 6), 'n': 4, 'k': 3
	}

	sys = System('mine')
	sys.fit(params, X_train, Y_train)
	sys.predict(X_test)
	print(sys.scores(Y_test))
