from utils import *

def check(df):
	if not 'label' in df.columns: raise "Please set label_col"

def prepare(name, *, label_col = None):
	df = pd.read_csv(full_fpath(name))

	if not label_col is None: df.rename(columns = {label_col: 'label'}, inplace = True)
	#TODO: Change label_col from any type to numbers
	df = df.select_dtypes(include = np.number)

	check(df)

prepare('kaggle')
