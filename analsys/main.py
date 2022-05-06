from utils import *
from system import SkSystem

def precision(results, expected):
	return (expected == results).sum() / expected.shape[0]

if __name__ == '__main__':
	k, usePca, n, niters, pct = 5, False, None, None, 0.4
	fpath = '../data/kaggle/small'
	data = split_data(fpath, pct)
	_, _, _, expected = data

	sys = SkSystem(data, k, usePca, n) #A: Run sklearn code
	_, results = run(fpath, k, usePca, n, niters) #A: Run c++ code

	print(precision(results, expected), precision(sys.results, expected))
