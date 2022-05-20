from system import Systems

import plot 
from utils import *
import gzip

class ExpRanges:
	def __init__(self, dataset, name, *, dataSz = None, frac = [0.6], threshold = [0], maxIters = [np.nan], n = [0], k = [10]): #U: Runs the program with different values 
		print(f'ExpRanges START {name}, {dataset}')
		self.dataset = dataset
		self.name = name
		self.data_df = pd.read_csv(full_fpath(dataset))
		self.path = exp_path(dataset, 'expRanges', name)
		self.fout_fpath = df_fpath(self.path, 'df') #TODO: ../data/../data
		self.ranges = {
			'dataSz': dataSz if not dataSz is None else [len(self.data_df)],
			'frac': frac,
			'threshold': threshold,
			'maxIters': maxIters,
			'n': n,
			'k': k,
		}

		self.run()
		self.read()
		self.plot()

	def run(self):
		systems = Systems(self.data_df)

		lastRange = 0
		try: df = pd.read_csv(self.fout_fpath) #A: Read previous df and skip ahead
		except FileNotFoundError:	print(f'ExpRanges RUN df not found, starting from scratch')
		else:
			lastRange = len(df) / 2 #NOTE: /2 because we're adding two rows per iteration #TODO: Get it from systems
			print(f'ExpRange {self.name} RUN continuing from {lastRange}')
		with gzip.open(self.fout_fpath, 'a') as fout:
			first = True
			for params in ( #A: For every combination of ranges
					{
						'dataSz': dataSz, 'frac': frac, 'threshold': threshold,
						'maxIters': maxIters, 'n': n,
						'k': k
					}
				for i, (dataSz, frac, threshold, maxIters, n, k) in enumerate(itt.product(
					self.ranges['dataSz'], self.ranges['frac'], self.ranges['threshold'],
					self.ranges['maxIters'], self.ranges['n'],
					self.ranges['k']
				)) if (
					(i >= lastRange) and #A: Skip combinations already present in df 
					(n <= min(dataSz * frac, 784)) and #SEE: https://stackoverflow.com/a/51041152
					(k <= (dataSz * frac))
				)
			):
				print(f'ExpRanges {self.name} RUN {params}')
				pd.DataFrame(systems.run(params)).to_csv( #A: Append this data to the df
					fout,
					index = False, mode = 'a', 
					header = first #A: Only add header on the first time
				) 
				first = False
		#TODO: Can this be optimized?

	def read(self):
		self.df = pd.read_csv(self.fout_fpath) #A: Read from the full df #TODO: confusion_matrix not read correctly 
		self.df['predictTime'] = np.log10(self.df['predictTime']) + 3 #A: Log and in milliseconds
		self.df['pcaTime'] = np.log10(self.df['pcaTime']) + 3
		self.df['trainSz'] = self.df['dataSz'] * self.df['frac']
		self.df['testSz'] = self.df['dataSz'] - self.df['trainSz']

	def plot(self):
		print(f'ExpRanges {self.name} PLOT')
		self.plotScores()
		self.plotTimes()


	def plotScores(self, *, scores = ['accuracy', 'cohenKappa', 'f1', 'precision', 'recall'], xy = [('k', 'n')]):
		for col, (x, y) in itt.product(scores, xy):
			plot.heatmap(
				self.df[(self.df['dataSz'] == 40000) & (self.df['frac'] == 0.6)],
				x, y, col,
				self.path
			)
	
	def plotTimes(self, *, cols = [
		('dataSz', 'k', 'predictTime'),
		('dataSz', 'n', 'predictTime'),
		('dataSz', 'n', 'pcaTime'),
	]):
		for x, y, col in cols:
			plot.heatmap(
				self.df,
				x, y, col,
				self.path
			)

if __name__ == '__main__':
	dataset = 'kaggle'
	ExpRanges(
		dataset, 'tiny',
		**{
			'dataSz': [20000],
			'n': [-1, 0, 1],
			'k': [1, 50, 100],
		}
	)
	ExpRanges(
		dataset, 'small',
		**{
			'dataSz': range(10000, 42001, 10000),
			'maxIters': [0, 10, 100, 1000, np.nan],
			'frac': [0.1, 0.5, 0.9],
			'n': [-1] + list(range(0, 50, 5)),
			'k': list(range(1, 10)) + list(range(10, 50 + 1, 10)),
		}
	)
	ExpRanges(
		dataset, 'big',
		**{
			'dataSz': range(1000, 42001, 1000),
			'maxIters': [0, 10, 100, 1000, np.nan],
			'frac': np.arange(0.2, 1, 0.2),
			'n': range(-1, 100 + 1),
			'k': range(1, 100 + 1),
		}
	)
	ExpRanges(
		dataset, 'full',
		**{
			'dataSz': range(100, 42001, 100),
			'frac': np.arange(0.1, 1, 0.1),
			'maxIters': list(range(0, 1000)) + [np.nan],
			'n': [-1] + list(range(0, 784 + 1)),
			'k': range(1, 42001),
		}
	)
