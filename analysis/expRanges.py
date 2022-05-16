from system import Systems

from plot import *
from utils import *

class ExpRanges:
	def __init__(self, name, ranges): #U: Runs the program with different values for k, a, 
		print(f'ExpRanges START name {name}, ranges {ranges}')
		self.name = name
		self.data_df = pd.read_csv(full_fpath(name))
		self.path = exp_path(name, 'expRanges')
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
				{'dataSz': dataSz, 'frac': frac, 'maxIters': maxIters, 'n': n, 'k': k}
			for i, (dataSz, frac, maxIters, n, k) in enumerate(itt.product(self.ranges['dataSz'], self.ranges['frac'], self.ranges['maxIters'], self.ranges['n'], self.ranges['k']))
			if i >= lastRange #A: Skip combinations already present in df #TODO: Can this be optimized?
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
		self.df['predictTime'] = np.log10(self.df['predictTime'])
		self.df['pcaTime'] = np.log10(self.df['pcaTime'])

	def plot(self):
		print('ExpRanges PLOT')
		df = self.df[(self.df['whose'] == 'mine') & (self.df['dataSz'] == 5000)]
		print(df[(df['k'] == 10) & (df['n'] == 1)]['predictTime'])
		for score in (col for col in df.columns if not col in ['k', 'n', 'whose', 'confusionMatrix', 'dataSz', 'testSz', 'maxIters']):
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
	name = 'kaggle'
	ExpRanges(
		name,
		# {'k': range(10, 51, 10), 'n': range(0, 50 + 1, 5), 'dataSz': range(1000, 5001, 1000), 'maxIters': list(range(0, 10)) + [np.pow(10, i) for i in range(1, 7)], 'frac': [0.4]},
		{'k': range(10, 51, 20), 'n': range(0, 6), 'dataSz': [5000], 'maxIters': [1000000], 'frac': [0.4]},
	)
