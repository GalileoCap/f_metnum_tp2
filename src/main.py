from system import Systems

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

	params = (1, 2, 1000000)

	sys = Systems()
	sys.fit(params, X_train, Y_train)
	sys.predict(X_test)
	print(sys.scores(Y_test))
