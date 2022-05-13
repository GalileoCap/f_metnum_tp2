from system import *

from plot import *
from utils import *

class ExpRanges:
	def __init__(self, name, ranges, mode = 'replace'): #U: Runs the program with different values for k, a, 
		print(f'ExpRanges START name {name} ranges {ranges}, mode {mode}')
		self.name = name
		self.path = exp_path(name, 'expRanges')
		self.df_fpath = df_fpath(self.path, 'df') #TODO: Rename
		self.tmpdf_fpath = df_fpath(self.path, '.tmpdf') #TODO: Rename
		self.ranges = ranges
		self.mode = mode

		if self.mode == 'read': self.read()
		else: self.run()
		self.plot()

	def run(self):
		print(f'ExpRanges RUN')

		startRanges = {'train': 0, 'k': 0, 'n': 0}
		if self.mode == 'continue': #A: Read tmpdf and skip ahead
			try:
				tmpdf = pd.read_csv(self.tmpdf_fpath)
			except FileNotFoundError:
				print(f'ExpRanges RUN tmpdf not found, starting from scratch')
			else:
				startRanges.update({
					'train': len(tmpdf['trainSz'].unique()),
					# 'k': len(tmpdf['k'].unique()),
					# 'n': len(tmpdf['n'].unique())
				})
				#TODO: More depth to continue
				print(f'ExpRanges RUN continuing from {startRanges}')
		elif os.path.exists(self.tmpdf_fpath): os.remove(self.tmpdf_fpath) #A: Delete tmpdf to start from scratch

		for train_sz in self.ranges['train'][startRanges['train']:]:
			print(f'ExpRanges run train_sz {train_sz}')
			frac = 0.4 #TODO: Change frac
			split_data(self.name, self.path, frac = frac, subset = ceil(train_sz / frac)) #A: Split the data with a subset the correct length for this frac 

			rows = []
			for k in self.ranges['k'][startRanges['k']:]:
				for n in self.ranges['n'][startRanges['n']:]:
					params = (k, n > 0, n, 1000000) #TODO: Change niter
					rows += self.run_inst(params)
			pd.DataFrame(rows).to_csv( #A: Append this data to the tmpdf
				self.tmpdf_fpath,
				index = False, mode = 'a', 
				header = not os.path.exists(self.tmpdf_fpath)
			) 
		#TODO: Can this be optimized using generators?
		self.df = pd.read_csv(self.tmpdf_fpath) #A: Read from the full tmpdf
		self.df.to_csv(self.df_fpath, index = False) #A: Write the actual df
		os.remove(self.tmpdf_fpath) #A: Delete the tmpdf #TODO: Why?

	def run_inst(self, params):
		mySys = MySystem(params, self.path) #A: Run c++ code
		skSys = SkSystem(params, self.path) #A: Run sklearn code
		return [mySys.metrics.copy(), skSys.metrics.copy()] #A: Add k and n to each row #TODO: Do I have to copy?

	def read(self): #U: Tries to read previous results
		try:
			self.df = pd.read_csv(self.df_fpath)
		except FileNotFoundError: #A: If the file doesn't exist we need to replace anyways
			print('ExpRanges ERROR file not found, replacing anyways')
			self.run()
		else:
			print('ExpRanges READ')
			#TODO: confusion_matrix not read correctly
			#TODO: Check if read correct file
			pass

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
		# {'k': range(10, 51, 10), 'n': range(0, 6 + 1), 'train': range(1000, 10001, 1000)},
		# mode = 'continue',
	)
