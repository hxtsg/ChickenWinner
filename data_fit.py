import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor as xgb

def load_data( INPUT_PATH ):
	data = pd.read_csv( INPUT_PATH )
	print( data.head() )
	print('Data Loaded')
	return data

def FeatureSelection( data ):
	SELECTED_NUM = 50000

	data = data[ data[ 'matchType' ] == 'solo-fpp' ]
	data.drop( [ 'Id', 'groupId', 'matchId', 'matchType' ], axis=1 ,inplace=True )
	data.dropna( inplace=True,axis=0 )

	data.reset_index( drop=True, inplace=True )

	# distance standardize
	data[ 'swimDistance' ] = ( data[ 'swimDistance' ] - data[ 'swimDistance' ].mean() ) / data[ 'swimDistance' ].std()
	data[ 'walkDistance' ] = ( data[ 'walkDistance' ] - data[ 'walkDistance' ].mean() ) / data[ 'walkDistance' ].std()
	data[ 'rideDistance' ] = ( data[ 'rideDistance' ] - data[ 'rideDistance' ].mean() ) / data[ 'rideDistance' ].std()

	# 
	data[ 'damageDealt' ] = ( data[ 'damageDealt' ] - data[ 'damageDealt' ].mean() ) / data[ 'damageDealt' ].std()
	data[ 'killPoints' ] = ( data[ 'killPoints' ] - data[ 'killPoints' ].mean() ) / data[ 'killPoints' ].std()
	data[ 'longestKill' ] = ( data[ 'longestKill' ] - data[ 'longestKill' ].mean() ) / data[ 'longestKill' ].std()
	data[ 'rankPoints' ] = ( data[ 'rankPoints' ] - data[ 'rankPoints' ].mean() ) / data[ 'rankPoints' ].std()
	data[ 'winPoints' ] = ( data[ 'winPoints' ] - data[ 'winPoints' ].mean() ) / data[ 'winPoints' ].std()



	data = data[ : SELECTED_NUM ]

	y = data['winPlacePerc']
	X = data.drop( [ 'winPlacePerc' ], axis=1 )
	return X, y


def plot_box( data ):
	input_feature = 'headshotKills'
	output_feature = 'winPlacePerc'

	X = pd.DataFrame( data = [ x for x in data[ input_feature ] ], columns = [ input_feature ] )
	full = pd.concat( [ X, data[ output_feature ] ], axis = 1 )
	plt.figure( figsize=(15,8) )
	sns.boxplot( full[ input_feature ], full[ output_feature ] )
	plt.show()

	return

def get_scores( model, X_train, y_train, X_test, y_test ):
	pred_train = model.predict( X_train )
	pred_test  = model.predict( X_test )

	r2_train =  r2_score( pred_train, y_train )
	mse_train = mean_squared_error( pred_train, y_train )

	r2_test = r2_score( pred_test, y_test )
	mse_test = mean_squared_error( pred_test, y_test )

	pred_test = model.predict( X_test )

	error_rate = np.sum( np.abs( pred_test - y_test.values ) ) * 1.0 / pred_test.shape[0]
	print( 'R2 Score Train:{:.6f}'.format( r2_train ) )
	print( 'MSE Score Train:{:.6f}' .format( mse_train ))
	print( 'R2 Score Test:{:.6f}'.format( r2_test ) )
	print( 'MSE Score Test:{:.6f}' .format( mse_test ))
	# print( error_rate )
	return error_rate

def Get_xgboostScore( X, y ):
	N_ESTIMATORS = 300
	CV_FOLD = 5
	loss_list = []
	for i in range( CV_FOLD ):
		test_idx_start = int( X.shape[0] / CV_FOLD ) * i
		test_idx_end   = int( X.shape[0] / CV_FOLD ) * ( i + 1 )
		X_train = pd.concat( [ X[ :test_idx_start ], X[ test_idx_end: ] ], axis=0 )
		y_train = pd.concat( [ y[ :test_idx_start ], y[ test_idx_end: ] ], axis=0 )
		X_test  = X[ test_idx_start:test_idx_end ]
		y_test  = y[ test_idx_start:test_idx_end ]
		# print( X_train.shape, y_train.shape, X_test.shape, y_test.shape )
		model = xgb( max_depth = 5, n_estimators = N_ESTIMATORS )
		model.fit( X_train, y_train )

		loss_list.append(get_scores( model, X_train, y_train, X_test, y_test))

	print( 'Average Error:{:.6f}'.format( np.mean( loss_list ) ) )
	return

def GetTrainingScores( X, y ):
	Get_xgboostScore( X, y )


def main():
	data = load_data( '../input/train_V2.csv' )
	# plot_box( data )
	X, y = FeatureSelection( data )
	GetTrainingScores( X, y )

	return




if __name__ == '__main__':
	main()