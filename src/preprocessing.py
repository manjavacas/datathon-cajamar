import math, os

import numpy as np
import pandas as pd


def preprocess(file: str, train: bool) -> pd.DataFrame:
    df = pd.read_csv(file, index_col=0, sep='|')
    df.fillna(method='ffill', inplace=True)	# Rellena campos nulos aplicando el método ffill

    df['MAXBUILDINGFLOOR'] = df['MAXBUILDINGFLOOR'].astype(np.int64)

    # Conversión de CADASTRALQUALITYID en números enteros
    df['CADASTRALQUALITYID'].replace(['A','B','C',1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9,10,11,12], inplace=True)
    df['CADASTRALQUALITYID'] = df['CADASTRALQUALITYID'].astype(np.int64)

    # Aumento de datos con diversos cálculos que creemos relevantes
    # Medias
    df['mean_red'] = df.loc[:, 'Q_R_4_0_0':'Q_R_4_1_0'].mean(axis=1)
    df['mean_green'] = df.loc[:, 'Q_G_3_0_0':'Q_G_3_1_0'].mean(axis=1)
    df['mean_blue'] = df.loc[:, 'Q_B_2_0_0':'Q_B_2_1_0'].mean(axis=1)
    df['mean_nir'] = df.loc[:, 'Q_NIR_8_0_0':'Q_NIR_8_1_0'].mean(axis=1)

    # Desviaciones típicas
    df['std_red'] = df.loc[:, 'Q_R_4_0_0':'Q_R_4_1_0'].std(axis=1)
    df['std_green'] = df.loc[:, 'Q_G_3_0_0':'Q_G_3_1_0'].std(axis=1)
    df['std_blue'] = df.loc[:, 'Q_B_2_0_0':'Q_B_2_1_0'].std(axis=1)
    df['std_nir'] = df.loc[:, 'Q_NIR_8_0_0':'Q_NIR_8_1_0'].std(axis=1)

    # Máximos
    df['max_red'] = df.loc[:, 'Q_R_4_0_0':'Q_R_4_1_0'].max(axis=1)
    df['max_green'] = df.loc[:, 'Q_G_3_0_0':'Q_G_3_1_0'].max(axis=1)
    df['max_blue'] = df.loc[:, 'Q_B_2_0_0':'Q_B_2_1_0'].max(axis=1)
    df['max_nir'] = df.loc[:, 'Q_NIR_8_0_0':'Q_NIR_8_1_0'].max(axis=1)

    # Mínimos
    df['min_red'] = df.loc[:, 'Q_R_4_0_0':'Q_R_4_1_0'].min(axis=1)
    df['min_green'] = df.loc[:, 'Q_G_3_0_0':'Q_G_3_1_0'].min(axis=1)
    df['min_blue'] = df.loc[:, 'Q_B_2_0_0':'Q_B_2_1_0'].min(axis=1)
    df['min_nir'] = df.loc[:, 'Q_NIR_8_0_0':'Q_NIR_8_1_0'].min(axis=1)

    return df