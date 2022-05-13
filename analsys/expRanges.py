from system import *

from plot import *
from utils import *

class ExpRanges:
	def __init__(self, name, ranges): #U: Runs the program with different values for k, a, 
		print(f'ExpRanges START name {name} ranges {ranges}')
		self.name = name
		self.path = exp_path(name, 'expRanges')
		self.df_fpath = df_fpath(self.path, 'df') 
		self.ranges = ranges

		self.run()
		self.plot()

	def run(self):
		lastRange = 0
		try: df = pd.read_csv(self.df_fpath) #A: Read previous df and skip ahead
		except FileNotFoundError:	print(f'ExpRanges RUN df not found, starting from scratch')
		else:
			lastRange = len(df) / 2 #NOTE: /2 because we're adding two rows per iteration
			print(f'ExpRange RUN continuing from {lastRange}')

		for (train_sz, k, n) in ( #A: For every combination of ranges
			ranges
			for i, ranges in enumerate(itt.product(self.ranges['train'], self.ranges['k'], self.ranges['n']))
			if i >= lastRange #A: Skip combinations already present in df #TODO: Can this be optimized?
		):
			print(f'ExpRanges RUN train_sz {train_sz}, k {k}, n {n}')
			params, frac = (k, n > 0, n, 1000000), 0.4 #TODO: Change niter and frac
			split_data(self.name, self.path, frac = frac, subset = ceil(train_sz / frac)) #A: Split the data with a subset the correct length for this frac 
			pd.DataFrame(self.run_inst(params)).to_csv( #A: Append this data to the df
				self.df_fpath,
				index = False, mode = 'a', 
				header = not os.path.exists(self.df_fpath) #A: Only add header on the first time
			) 
		#TODO: Can this be optimized using generators?
		self.df = pd.read_csv(self.df_fpath) #A: Read from the full df #TODO: confusion_matrix not read correctly #TODO: Check if read correct file
		#TODO: Separate running the experiment from managing the df

	def run_inst(self, params):
		mySys = MySystem(params, self.path) #A: Run c++ code
		skSys = SkSystem(params, self.path) #A: Run sklearn code
		return [mySys.metrics.copy(), skSys.metrics.copy()] #A: Add k and n to each row #TODO: Do I have to copy?

	def plot(self): #TODO: Make more specific
		print('ExpRanges PLOT')
		for y in ['accuracy']:
			fig = go.Figure(data = go.Heatmap(
				x = self.df['k'],
				y = self.df['n'],
				z = self.df['guessTime'],
				colorbar = dict(title = 'Tiempo (μs)'),
			))
			# fig.update_layout(
				# # title = f'Relación entre el tiempo resolver y el tiempo para calcular LU ({reps} reps)',
				# # xaxis_title = 'Cantidad de ',
				# # yaxis_title = 'Cantidad de ángulos',
			# )
			# fig.write_image(img_fpath(f'{fpath}.heatmap'))
			fig.write_image(img_fpath(f'{self.path}/{y}.heatmap'))
		# for y in ['accuracy']:
			# for x in ['k', 'n']:
				# fig = go.Figure()
				# for i in (self.ranges['n'] if x == 'k' else self.ranges['k']): #A: Only trace some n's and k's #TODO: Change range
					# df = self.df[(self.df['whose'] == 'mine') & (self.df['n' if x == 'k' else 'k'] == i)] #TODO: Different whose
					# fig.add_trace(goScatter(df, x, y, f'{"α" if x == "k" else "k"} = {i}'))
					# fig.update_layout(
						# # legend_title = 'Cantidad de componentes', #TODO
						# xaxis_title = x,
						# yaxis_title = y,
					# )
				# fig.write_image(img_fpath(f'{self.path}/{y}.{x}'))

if __name__ == '__main__':
	name = 'kaggle'
	ExpRanges(
		name,
		{'k': [1], 'n': [0, 1], 'train': [1, 2, 3]},
		# {'k': range(10, 51, 10), 'n': range(0, 6 + 1), 'train': range(1000, 10001, 1000)}
	)
