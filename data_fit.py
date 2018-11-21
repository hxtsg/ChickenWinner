import numpy as np
import pandas as pd


def load_data( INPUT_PATH ):
	data = pd.read_csv( INPUT_PATH )
	print( data.head() )
	return


def main():
	data = load_data( '../input/train_V2.csv' )
	return




if __name__ == '__main__':
	main()