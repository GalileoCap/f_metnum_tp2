from system import Systems

from plot import *
from utils import *

class ExpRanges:
	def __init__(self, dataset, name, ranges): #U: Runs the program with different values for k, a, 
		print(f'ExpRanges START dataset {dataset}, ranges {ranges}')
		self.dataset = dataset
		self.name = name
		self.data_df = pd.read_csv(full_fpath(dataset))
		self.path = exp_path(dataset, 'expRanges', name)
		self.fout_fpath = df_fpath(self.path, 'df') 
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
		self.df = pd.read_csv(self.fout_fpath) #A: Read from the full df #TODO: confusion_matrix not read correctly #TODO: Check if read correct file
		self.df['predictTime'] = np.log10(self.df['predictTime']) + 3
		self.df['pcaTime'] = np.log10(self.df['pcaTime']) + 3

	def plot(self):
		print('ExpRanges PLOT')
		df = self.df[(self.df['whose'] == 'mine')]
		print(self.df.max())
		for score in (col for col in df.columns if not col in ['k', 'n', 'whose', 'confusionMatrix', 'dataSz', 'testSz', 'maxIters', 'frac', 'threshold']):
			fig = go.Figure(data = go.Heatmap(
				x = df['k'],
				y = df['n'],
				z = df[score],
				colorbar = dict(title = score),
			))
			fig.update_layout(
				# title = f'Relación entre el tiempo resolver y el tiempo para calcular LU ({reps} reps)',
				xaxis_title = 'k',
				yaxis_title = 'n',
			)
			fig.write_image(img_fpath(f'{self.path}/{score}.heatmap'))

if __name__ == '__main__':
	dataset = 'kaggle'
	ExpRanges(
		dataset, 'small',
		{
			'dataSz': range(10000, 42001, 10000),
			'frac': [0.1, 0.5, 0.9],
			'threshold': [0, 128, 256],
			'maxIters': [np.nan],
			'n': range(0, 784 + 1, 5),
			'k': list(range(1, 10)) + list(range(10, 51, 10)),
		}
	)
	ExpRanges(
		dataset, 'full',
		{
			'dataSz': range(1000, 42001, 1000),
			'frac': np.arange(0.1, 1, 0.1),
			'threshold': [0, 64, 128, 192, 256],
			'maxIters': list(range(0, 10)) + [1000, np.nan],
			'n': range(0, 784 + 1),
			'k': list(range(1, 5)) + list(range(5, 51, 5)),
		}
	)
