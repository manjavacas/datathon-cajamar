import sys, os

import preprocessing

from joblib import load


ESTIMAR_PATH = os.path.join('.', 'data', 'Estimar_UH2020.txt')
MODEL1 = os.path.join('.', 'models', 'model1.joblib')
MODEL2 = os.path.join('.', 'models', 'model2.joblib')

CLASES = ['AGRICULTURE', 'INDUSTRIAL', 'OFFICE', 'OTHER', 'PUBLIC', 'RESIDENTIAL', 'RETAIL']


def model1(df):
	clf = load(MODEL1)
	
	df['RESIDENTIAL_PREDICTED'] = clf.predict(df)
	
	df_residential = df.drop(df[df.RESIDENTIAL_PREDICTED == 0].index)
	df_residential['CLASE'] = 'RESIDENTIAL'
	df_residential.drop('RESIDENTIAL_PREDICTED', axis=1, inplace=True)

	df_noResidential = df.drop(df[df.RESIDENTIAL_PREDICTED == 1].index)
	df_noResidential.drop('RESIDENTIAL_PREDICTED', axis=1, inplace=True)

	return df_residential, df_noResidential


def model2(df):
	clf = load(MODEL2)

	df['CLASE'] = clf.predict(df)
	df['CLASE'].replace([i for i in range(len(CLASES))], CLASES, inplace=True)

	return df


def main():
	selected_features = [
		'Q_R_4_0_0', 'Q_R_4_0_1', 'Q_R_4_0_2', 'Q_R_4_0_3', 'Q_R_4_0_4', 'Q_R_4_0_5', 'Q_R_4_0_6', 'Q_R_4_0_7','Q_R_4_0_8', 'Q_R_4_0_9', 'Q_R_4_1_0', 
		'Q_G_3_0_0', 'Q_G_3_0_1', 'Q_G_3_0_2', 'Q_G_3_0_3', 'Q_G_3_0_4', 'Q_G_3_0_5', 'Q_G_3_0_6', 'Q_G_3_0_7', 'Q_G_3_0_8', 'Q_G_3_0_9', 'Q_G_3_1_0', 
		'Q_B_2_0_0', 'Q_B_2_0_1', 'Q_B_2_0_2', 'Q_B_2_0_3', 'Q_B_2_0_4', 'Q_B_2_0_5', 'Q_B_2_0_6', 'Q_B_2_0_7', 'Q_B_2_0_8', 'Q_B_2_0_9', 'Q_B_2_1_0', 
		'Q_NIR_8_0_0', 'Q_NIR_8_0_1', 'Q_NIR_8_0_2', 'Q_NIR_8_0_3', 'Q_NIR_8_0_4', 'Q_NIR_8_0_5', 'Q_NIR_8_0_6', 'Q_NIR_8_0_7', 'Q_NIR_8_0_8', 'Q_NIR_8_0_9', 'Q_NIR_8_1_0',
		
		'AREA',
		'GEOM_R1', 'GEOM_R2', 'GEOM_R3', 'GEOM_R4',
		'CONTRUCTIONYEAR',
		'MAXBUILDINGFLOOR',
		'CADASTRALQUALITYID'
	]

	df = preprocessing.preprocess(ESTIMAR_PATH, train=False)

	df_estimar = df[selected_features]
	
	residential, noResidential = model1(df_estimar)

	noResidential = model2(noResidential)

	residential = residential.append(noResidential)

	residential['CLASE'].to_csv('ESI_Anacongas.txt', sep='|')


if __name__ == '__main__':
	main()