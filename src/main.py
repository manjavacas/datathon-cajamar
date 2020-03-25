import sys, os

import preprocessing

from joblib import load


ESTIMAR_PATH = os.path.join('.', 'data', 'Estimar_UH2020.txt')
MODEL1 = os.path.join('.', 'models', 'model1.joblib')
MODEL2 = os.path.join('.', 'models', 'model2.joblib')

CLASES = ['AGRICULTURE', 'INDUSTRIAL', 'OFFICE', 'OTHER', 'PUBLIC', 'RESIDENTIAL', 'RETAIL']

# Características seleccionadas para el primer modelo
SELECTED_FEATURES_1 = [
  	'mean_red', 'mean_green', 'mean_blue', 'mean_nir',

  	'AREA',
  	'GEOM_R1', 'GEOM_R2', 'GEOM_R3', 'GEOM_R4',
  	'CONTRUCTIONYEAR',
  	'MAXBUILDINGFLOOR',
  	'CADASTRALQUALITYID'
	]

# Características seleccionadas para el segundo modelo
SELECTED_FEATURES_2 = [
  	'X','Y',

  	'max_red', 'max_green', 'max_blue', 'max_nir',

  	'AREA',
  	'GEOM_R1', 'GEOM_R2', 'GEOM_R3', 'GEOM_R4',
  	'CONTRUCTIONYEAR',
  	'MAXBUILDINGFLOOR',
  	'CADASTRALQUALITYID'
	]


def model1(df):
	'''
	Primer modelo. Clasificador Residencial vs No residencial
	Input: dataset completo
	Output: (dataset de residenciales, dataset de no residenciales)
	'''
	clf = load(MODEL1)
	
	df['RESIDENTIAL_PREDICTED'] = clf.predict(df[SELECTED_FEATURES_1])
	
	df_residential = df.drop(df[df.RESIDENTIAL_PREDICTED == 0].index)
	df_residential['CLASE'] = 'RESIDENTIAL'
	df_residential.drop('RESIDENTIAL_PREDICTED', axis=1, inplace=True)

	df_noResidential = df.drop(df[df.RESIDENTIAL_PREDICTED == 1].index)
	df_noResidential.drop('RESIDENTIAL_PREDICTED', axis=1, inplace=True)

	return df_residential, df_noResidential


def model2(df):
	'''
	Segundo modelo. Clasificador general
	Input: dataset con instancias predichas como no residenciales
	Output: dataset de entrada con sus predicciones
	'''
	clf = load(MODEL2)

	df['CLASE'] = clf.predict(df[SELECTED_FEATURES_2])
	df['CLASE'].replace([i for i in range(len(CLASES))], CLASES, inplace=True)

	return df


def main():
	df = preprocessing.preprocess(ESTIMAR_PATH, train=False)
	
	residential, noResidential = model1(df)

	noResidential = model2(noResidential)

	residential = residential.append(noResidential)

	residential['CLASE'].to_csv('ESI_Anacongas.txt', sep='|')


if __name__ == '__main__':
	main()