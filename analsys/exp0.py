import plot
import plotly.graph_objects as go
import pandas as pd
from system import *
from utils import *

class Exp0:
	def __init__(self, range_k, range_n, fpath, replace): #U: Runs the program with different values for k and a #TODO: Rename
		print(f'Exp0 START range_k {range_k}, range_n {range_n}, fpath {fpath}, replace {replace}')
		self.range_k = range_k
		self.range_n = range_n
		self.fpath = fpath
		self.df_fpath = pandas_fpath(f'{fpath}.exp0')
		self.replace = replace

		if replace: self.run()
		else: self.read()

		self.plot()

	def run(self):
		print(f'Exp0 RUN')
		scores = []
		for k in self.range_k:
			for n in self.range_n:
				params = (k, n > 0, n, 1000000) #TODO: Change niters
				scores += self.run_inst(params)
							
		self.df = pd.DataFrame(scores)
		self.df.to_csv(self.df_fpath, index = False) 

	def run_inst(self, params):
		mySys = MySystem(params, self.fpath) #A: Run c++ code
		skSys = SkSystem(params, self.fpath) #A: Run sklearn code

		(k, _, n, _) = params
		scores = [mySys.scores.copy(), skSys.scores.copy()] #A: Add k and n to each row
		for s in scores: s.update({'k': k, 'n': n})
		return scores 

	def read(self): #U: Tries to read previous results
		try:
			self.df = pd.read_csv(self.df_fpath)
		except FileNotFoundError: #A: If the file doesn't exist we need to replace anyways
			print('Exp0 ERROR file not found, replacing anyways')
			self.run()
		else:
			#TODO: confusion_matrix not read correctly
			#TODO: Check if read correct file
			pass

	def plot(self):
		for y in ['accuracy']:
			for x in ['k', 'n']:
				fig = go.Figure()
				for i in (self.range_n if x == 'k' else self.range_k): #A: Only trace some n's and k's #TODO: Change range
					df = self.df[(self.df['whose'] == 'mine') & (self.df['n' if x == 'k' else 'k'] == i)] #TODO: Different whose
					fig.add_trace(plot.goScatter(df, x, y, f'{"α" if x == "k" else "k"} = {i}'))
					fig.update_layout(
						# legend_title = 'Cantidad de componentes', #TODO
						xaxis_title = x,
						yaxis_title = y,
					)
				fig.write_image(img_fpath(f'{self.fpath}.exp0.{y}.{x}'))

if __name__ == '__main__':
	fpath = '../data/kaggle/small'
	split_data(0.4, fpath)
	Exp0(range(10, 50 + 1, 10), range(0, 10 + 1), fpath, True)
