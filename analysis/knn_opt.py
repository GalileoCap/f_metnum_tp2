import numpy as np

def old_distances(X, Y):
	return np.array([
		[pow(np.linalg.norm(x - y), 2) for y in Y]
		for x in X
	])

def row_norm_sqrd(X):
	return np.array([
		pow(np.linalg.norm(x), 2)
		for x in X
	]).reshape(-1, 1)

def opt_distances(X, Y):
	XX = np.broadcast_to(row_norm_sqrd(X), (X.shape[0], Y.shape[0]))
	YY = np.broadcast_to(row_norm_sqrd(Y), (Y.shape[0], X.shape[0]))
	return -2 * np.matmul(X, Y.T) + XX + YY.T

def steps(X, Y, foo = False):
	_X, _Y = X.copy(), Y.copy()
	if foo: _X, _Y = Y.copy(), X.copy()
	print('0', _X, sep = '\n')
	_X = np.power(_X, 2)
	print('1', _X, sep = '\n')
	_X = np.sum(_X, 1)
	print('2', _X, sep = '\n')
	_X = _X.reshape(_X.shape[0], 1)
	print('3', _X, sep = '\n')
	_X = np.broadcast_to(_X, (_X.shape[0], _Y.shape[0]))
	return _X

if __name__ == '__main__':
	# x_elems, y_elems = 2, 2
	# X = np.random.rand(x_elems, 1) * 10	
	# Y = np.random.rand(y_elems, 1) * 5
	X = np.array([[0, 0], [0, 1]])
	Y = np.array([[0, 0], [1, 1]])
	if True:
		distances =	{
			'old': old_distances(X, Y),
			'opt': opt_distances(X, Y)
		}
		for k, dists in distances.items():
			print(k)
			for x in range(X.shape[0]):
				print(x, dists[x])

	# print('X', steps(X, Y), np.broadcast_to(row_norm_sqrd(X), (X.shape[0], Y.shape[0])), sep = '\n')
	# print('Y', steps(X, Y, True), np.broadcast_to(row_norm_sqrd(Y), (Y.shape[0], X.shape[0])), sep = '\n')
