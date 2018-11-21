import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data( INPUT_PATH ):
	data = pd.read_csv( INPUT_PATH )
	print( data.head() )
	return data

def plot_box( data ):
	input_feature = 'headshotKills'
	output_feature = 'winPlacePerc'

	X = pd.DataFrame( data = [ x for x in data[ input_feature ] ], columns = [ input_feature ] )
	full = pd.concat( [ X, data[ output_feature ] ], axis = 1 )
	plt.figure( figsize=(15,8) )
	sns.boxplot( full[ input_feature ], full[ output_feature ] )
	plt.show()

	return


def main():
	data = load_data( '../input/train_V2.csv' )
	plot_box( data )
	return




if __name__ == '__main__':
	main()