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

