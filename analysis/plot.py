import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils import *

def eigens(components, path):
	fig = make_subplots(
		rows = ceil(len(components) / 4), cols = 4,
		subplot_titles = [f'Componente #{i+1}' for i in range(len(components))]
	)
	for i, component in enumerate(components):
		fig.add_trace(
			px.imshow(np.array(component).reshape(28, 28)).data[0],
			row = (i // 4) + 1,
			col = (i % 4) + 1,
		)
	fig.update_layout( 
		coloraxis = px.imshow( #A: Grayscale #SEE: https://stackoverflow.com/questions/64268081/creating-a-subplot-of-images-with-plotly
			np.array(components[0]).reshape(28, 28),
			color_continuous_scale = 'gray'
		).layout.coloraxis, 
		coloraxis_showscale = False, 
	)
	fig.update_xaxes(showticklabels = False) #A: Hide ticks
	fig.update_yaxes(showticklabels = False) 
	fig.write_image(img_fpath(f'{path}/eigens'))

def heatmap(df, x, y, col, path):
	for whose in df['whose'].unique():
		_df = df[(df['whose'] == whose)]
		fig = go.Figure(go.Heatmap(
			x = _df[x].unique(),
			y =	_df[y].unique(),
			z = [
				[_df[(_df[x] == _x) & (_df[y] == _y)][col].mean()
				for _x in _df[x].unique()]
				for _y in _df[y].unique()
			],
			colorbar = dict(title = col),
		))
		fig.update_layout(
			# title = f'Relación entre el tiempo resolver y el tiempo para calcular LU ({reps} reps)',
			xaxis_title = x,
			yaxis_title = y,
		)
		fig.write_image(img_fpath(f'{path}/heatmap.{whose}.{x}.{y}.{col}'))

def scatter(df, x, col, path): #TODO: Figure out
	fig = go.Figure()
	for whose in df['whose'].unique():
		_df = df[(df['whose'] == whose)]
		fig.add_trace(go.Scatter(
			x = _df[x].unique(),
			y = [
				_df[(_df[x] == _x)][col].mean()
				for _x in _df[x].unique()
			],
			name = whose,
		))
	fig.update_layout(
		# title = f'Relación entre el tiempo resolver y el tiempo para calcular LU ({reps} reps)',
		xaxis_title = x,
		yaxis_title = col,
	)
	fig.write_image(img_fpath(f'{path}/scatter.{x}.{col}'))

