import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import numpy as np
from utils import *

def eigens(results, fpath):
	for k, (_, eigens, _) in results.items():
		fig = make_subplots(rows = ceil(len(eigens), 4), cols = 4, subplot_titles = [f'Componente #{i+1}' for i in range(len(eigens))])
		for i, eigen in enumerate(eigens):
			fig.add_trace(
				px.imshow(np.array(eigen).reshape(28, 28)).data[0],
				row = (i // 4) + 1,
				col = (i % 4) + 1,
			)
		fig.update_layout(
			coloraxis = px.imshow(np.array(eigens[0]).reshape(28, 28), color_continuous_scale='gray').layout.coloraxis, #SEE: https://stackoverflow.com/questions/64268081/creating-a-subplot-of-images-with-plotly
			coloraxis_showscale = False, 
		)
		fig.update_xaxes(showticklabels = False) # hide all the xticks
		fig.update_yaxes(showticklabels = False) # hide all the xticks
		fig.write_image(img_fpath(f'{fpath}.{k}.eigens'))

def go_scatter(df, x, y, name):
	return go.Scatter(
		x = df[x],
		y = df[y],
		name = name,
	)

def exp0(df, fpath):
	for y in ['accuracy']:
		for x in ['k', 'n']:
			fig = go.Figure()
			for i in (range(1, 11) if x == 'k' else range(10, 51, 10)): #A: Only trace some n's and k's #TODO: Change range
				_df = df[(df['whose'] == 'mine') & (df['n' if x == 'k' else 'k'] == i)] #TODO: Different whose
				fig.add_trace(go_scatter(_df, x, y, f'{"α" if x == "k" else "k"} = {i}'))
				fig.update_layout(
					# legend_title = 'Cantidad de componentes', #TODO
					xaxis_title = x,
					yaxis_title = y,
				)
			fig.write_image(img_fpath(f'{fpath}.exp0.{y}.{x}'))
