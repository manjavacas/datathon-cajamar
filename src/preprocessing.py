import math, os

import numpy as np
import pandas as pd


def preprocess(file: str, train: bool) -> pd.DataFrame:
	df = pd.read_csv(file, index_col=0, sep='|')
	df.dropna(inplace=True)

	df['MAXBUILDINGFLOOR'] = df['MAXBUILDINGFLOOR'].astype(np.int64)

	df['CADASTRALQUALITYID'].replace(['A','B','C',1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9,10,11,12], inplace=True)
	df['CADASTRALQUALITYID'] = df['CADASTRALQUALITYID'].astype(np.int64)

	# Means
	df['mean_red'] = df.loc[:, 'Q_R_4_0_0':'Q_R_4_1_0'].mean(axis=1)
	df['mean_green'] = df.loc[:, 'Q_G_3_0_0':'Q_G_3_1_0'].mean(axis=1)
	df['mean_blue'] = df.loc[:, 'Q_B_2_0_0':'Q_B_2_1_0'].mean(axis=1)
	df['mean_nir'] = df.loc[:, 'Q_NIR_8_0_0':'Q_NIR_8_1_0'].mean(axis=1)

	# Standard desviations
	df['std_red'] = df.loc[:, 'Q_R_4_0_0':'Q_R_4_1_0'].std(axis=1)
	df['std_green'] = df.loc[:, 'Q_G_3_0_0':'Q_G_3_1_0'].std(axis=1)
	df['std_blue'] = df.loc[:, 'Q_B_2_0_0':'Q_B_2_1_0'].std(axis=1)
	df['std_nir'] = df.loc[:, 'Q_NIR_8_0_0':'Q_NIR_8_1_0'].std(axis=1)

	# Maximums
	df['max_red'] = df.loc[:, 'Q_R_4_0_0':'Q_R_4_1_0'].max(axis=1)
	df['max_green'] = df.loc[:, 'Q_G_3_0_0':'Q_G_3_1_0'].max(axis=1)
	df['max_blue'] = df.loc[:, 'Q_B_2_0_0':'Q_B_2_1_0'].max(axis=1)
	df['max_nir'] = df.loc[:, 'Q_NIR_8_0_0':'Q_NIR_8_1_0'].max(axis=1)

	# Minimums
	df['min_red'] = df.loc[:, 'Q_R_4_0_0':'Q_R_4_1_0'].min(axis=1)
	df['min_green'] = df.loc[:, 'Q_G_3_0_0':'Q_G_3_1_0'].min(axis=1)
	df['min_blue'] = df.loc[:, 'Q_B_2_0_0':'Q_B_2_1_0'].min(axis=1)
	df['min_nir'] = df.loc[:, 'Q_NIR_8_0_0':'Q_NIR_8_1_0'].min(axis=1)

	df.drop(['X', 'Y'], axis=1, inplace=True)

	return df
