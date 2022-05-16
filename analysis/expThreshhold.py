from system import *

from plot import *
from utils import *

class ExpThreshold:
	def __init__(self, name, ranges): #U: Runs the program with different values for k, a, 
		print(f'ExpThreshold START name {name}, ranges {ranges}')
		self.name = name
		self.path = exp_path(name, 'expThreshold')
		self.data_fpath = df_fpath(self.path, 'data') 
		self.df_fpath = df_fpath(self.path, 'df') 
		self.ranges = ranges

		self.run()
		self.read()
		self.plot()

	def run(self):
		lastRange = 0
		try: df = pd.read_csv(self.df_fpath) #A: Read previous df and skip ahead
		except FileNotFoundError:	print(f'ExpThreshold RUN df not found, starting from scratch')
		else:
			lastRange = len(df) / 2 #NOTE: /2 because we're adding two rows per iteration
			print(f'ExpRange RUN continuing from {lastRange}')

		for (train_sz, k, n) in ( #A: For every combination of ranges
			ranges
			for i, ranges in enumerate(itt.product(self.ranges['train'], self.ranges['k'], self.ranges['n']))
			if i >= lastRange #A: Skip combinations already present in df #TODO: Can this be optimized?
		):
			print(f'ExpThreshold RUN train_sz {train_sz}, k {k}, n {n}')
			params, frac = (k, n > 0, n, 1000000), 0.4 #TODO: Change niter and frac
			split_data(self.name, self.path, frac = frac, subset = ceil(train_sz / frac)) #A: Split the data with a subset the correct length for this frac 
			pd.DataFrame(self.run_inst(params)).to_csv( #A: Append this data to the df
				self.df_fpath,
				index = False, mode = 'a', 
				header = not os.path.exists(self.df_fpath) #A: Only add header on the first time
			) 
		#TODO: Can this be optimized using generators?
		#TODO: Separate running the experiment from managing the df

	def run_inst(self, params):
		mySys = MySystem(params, self.path) #A: Run c++ code
		skSys = SkSystem(params, self.path) #A: Run sklearn code
		return [mySys.metrics.copy(), skSys.metrics.copy()] #A: Add k and n to each row #TODO: Do I have to copy?

	def read(self):
		self.df = pd.read_csv(self.df_fpath) #A: Read from the full df #TODO: confusion_matrix not read correctly #TODO: Check if read correct file
		self.df['guessTime'] = np.log10(self.df['guessTime'])
		self.df['pcaTime'] = np.log10(self.df['pcaTime'])

	def plot(self):
		print('ExpThreshold PLOT')
		_df = self.df[(self.df['whose'] == 'mine') & (self.df['trainSz'] == 5000)]
		print(_df[(_df['k'] == 10) & (_df['n'] == 1)]['guessTime'])
		for score in (col for col in _df.columns if not col in ['k', 'n', 'whose', 'confusionMatrix', 'trainSz', 'testSz', 'niters']):
			fig = go.Figure(data = go.Heatmap(
				x = _df['k'],
				y = _df['n'],
				z = _df[score],
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
	ExpThreshold(
		name,
		# {'k': range(10, 51, 10), 'n': range(0, 6 + 1), 'train': range(1000, 5001, 1000)},
		{'k': [10], 'n': range(0, 30 + 1), 'train': [5000]},
	)
