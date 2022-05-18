from system import Systems

import plot 
from utils import *

class ExpRanges:
	def __init__(self, dataset, name, ranges): #U: Runs the program with different values for k, a, 
		print(f'ExpRanges START dataset {dataset}, ranges {ranges}')
		self.dataset = dataset
		self.name = name
		self.data_df = pd.read_csv(full_fpath(dataset))
		self.path = exp_path(dataset, 'expRanges', name)
		self.fout_fpath = df_fpath(self.path, 'df') #TODO: ../data/../data
		self.ranges = ranges

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
			print(f'ExpRange RUN continuing from {lastRange}')
		for params in ( #A: For every combination of ranges
				{
					'dataSz': dataSz, 'frac': frac, 'threshold': threshold,
					'maxIters': maxIters, 'n': n,
					'k': k
				}
			for i, (dataSz, frac, threshold, maxIters, n, k) in enumerate(itt.product(
				self.ranges['dataSz'], self.ranges['frac'], self.ranges['threshold'],
				self.ranges['maxIters'], self.ranges['n'],
				self.ranges['k'])
			) if (
				(i >= lastRange) and #A: Skip combinations already present in df 
				(n <= min(dataSz * frac, 784)) #SEE: https://stackoverflow.com/a/51041152
			)
		):
			print(f'ExpRanges RUN params {params}')
			pd.DataFrame(systems.run(params)).to_csv( #A: Append this data to the df
				self.fout_fpath,
				index = False, mode = 'a', 
				header = not os.path.exists(self.fout_fpath) #A: Only add header on the first time
			) 
		#TODO: Can this be optimized?

	def read(self):
		self.df = pd.read_csv(self.fout_fpath) #A: Read from the full df #TODO: confusion_matrix not read correctly 
		self.df['predictTime'] = np.log10(self.df['predictTime']) + 3
		self.df['pcaTime'] = np.log10(self.df['pcaTime']) + 3

	def plot(self):
		print('ExpRanges PLOT')
		df = self.df[
			(self.df['threshold'] == 0) &
			(self.df['dataSz'] == 20000)
		]
		paramCols = ['k', 'n', 'whose', 'dataSz', 'maxIters', 'frac', 'threshold']
		skipCols = ['confusionMatrix']
		for col in df.columns: 
			if (col in paramCols) or (col in skipCols): continue
			for x, y in [('k', 'n')]:
				plot.heatmap(df, 'k', 'n', col, self.path)
			# for x in ['k', 'n']:
				# plot.scatter(df, x, col, self.path)

if __name__ == '__main__':
	dataset = 'kaggle'
	ExpRanges(
		dataset, 'small',
		ranges = {
			'dataSz': range(10000, 42001, 10000),
			'frac': [0.1, 0.5, 0.9],
			'threshold': [0],
			'maxIters': [np.nan],
			'n': [-1] + list(range(0, 50, 5)),
			'k': list(range(1, 10)) + list(range(10, 51, 10)),
		}
	)
	ExpRanges(
		dataset, 'full',
		ranges = {
			'dataSz': range(1000, 42001, 1000),
			'frac': np.arange(0.1, 1, 0.1),
			'threshold': [0],
			'maxIters': list(range(0, 10)) + [1000, np.nan],
			'n': [-1] + list(range(0, 784 + 1, 5)),
			'k': list(range(1, 5)) + list(range(5, 51, 5)),
		}
	)
