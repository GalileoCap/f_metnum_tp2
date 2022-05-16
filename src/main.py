import tp2
import numpy as np
import sklearn.decomposition 
# import plotly.figure_factory as ff
# import plotly.graph_objects as go
import plotly.express as px

def power_iteration(A, niter=10_000, eps=1e-6):
	a = 1
	v = np.random.rand(A.shape[0])
	for i in range(niter):
		v = np.dot(A, v) / np.linalg.norm(np.dot(A, v))
		a = np.dot(v.T, np.dot(A, v)) / pow(np.linalg.norm(v), 2)
		if np.allclose(np.dot(A, v), a * v, eps): 
			break
	return a, v

def eig(A, num=2, niter=10000, eps=1e-6):
	A = A.copy()
	eigenvalues = []
	eigenvectors = np.zeros((num, A.shape[1]))
	for i in range(num):
		a, v = power_iteration(A, niter, eps)
		A -= a * np.dot(v, v.T)
		eigenvalues.append(a)
		eigenvectors[i] = v.T
	return eigenvalues, eigenvectors

if __name__ == '__main__':
	# X = np.random.rand(2, 2) * 10
	X = np.array([
		[0., 0., 0],
		[0., 1., 1],
		[1., 0., 2],
		[1., 1., 3]
	])
	X, Y = X[:, :-1], X[:, -1]
	X -= np.mean(X, axis = 0)
	M = np.dot(X.T, X) / X.shape[0]
	print(f'X: {X}\nY: {Y} \nM: {M}')

	PCA = tp2.PCA(X.shape[1], 10000000); PCA.fit(M)
	X = PCA.transform(X)

	KNN = tp2.KNN(1); KNN.fit(X, Y)
	res = KNN.guess(X)
	print(res, sep = '\n')
