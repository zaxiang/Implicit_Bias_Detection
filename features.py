import pandas as pd
import numpy as np
import datetime
import ast

from matplotlib import pyplot as plt

def normalize(series):
    """
    normalize a series
    """

    array = np.array(series)
    array_norm = (array-np.min(array))/(np.max(array)-np.min(array))
    return pd.Series(array_norm)


def arrest_features_label(data):
    """
    feature engineering for arrest dataset
    with normalization
    """

    arrest = data.dropna(axis=0)

    #date (month, date), and time (hour,  minute)
    day = arrest['Arrest Date'].apply(lambda x: int(str(x)[8:10]))
    month = arrest['Arrest Date'].apply(lambda x: int(str(x)[5:7]))
    hour = arrest['Time'].apply(lambda x: int(str(int(x))[:-2]) if str(int(x))[:-2] != '' else 0)
    minute = arrest['Time'].apply(lambda x: int(str(int(x))[-2:]) if str(int(x))[:-2] != '' else 0)

    #location (Geographic Areas, latitude, longitude)
    area_ID = arrest['Area ID']
    latitude = arrest['Location'].apply(lambda x: float(ast.literal_eval(x)['latitude']))
    longitude = arrest['Location'].apply(lambda x: float(ast.literal_eval(x)['longitude']))

    #charge and arrest type
    dummies_arrest = pd.get_dummies(arrest['Arrest Type Code'])
    dummies_charge = pd.get_dummies(arrest['Charge Group Code'])

    #Age
    age = arrest['Age']
    #sex
    dummies_sex = pd.get_dummies(arrest['Sex Code'])
    #race
    dummies_race = pd.get_dummies(arrest['Descent Code'])

    need_normalize = [day, month, hour, minute, area_ID, latitude, longitude, age]
    for i in range(len(need_normalize)):
        need_normalize[i] = normalize(need_normalize[i])



    df_feature = pd.concat({'day':need_normalize[0], 'month':need_normalize[1], 
                            'hour':need_normalize[2], 'minute': need_normalize[3], 
                            'area':need_normalize[4], 
                            'lat':need_normalize[5], 'lon':need_normalize[6], 
                            'age':need_normalize[7]
                            },axis=1)

    dummies_arrest = dummies_arrest.reset_index()
    dummies_charge = dummies_charge.reset_index()
    dummies_sex = dummies_sex.reset_index()

    df_feature = df_feature.merge(dummies_arrest, left_index=True, right_index=True)
    df_feature = df_feature.merge(dummies_charge, left_index=True, right_index=True)
#     df_feature = df_feature.merge(dummies_sex, left_index=True, right_index=True) #removed sex feature

    df_feature = df_feature.drop(columns=['index_x', 'index_y']) #removed 'index' since no longer merging sex df

    return df_feature, arrest['Descent Code']


def crime_features(data):
    """
    feature engineering for crime dataset
    """

    crime = data[['Date Occurred', 'Time Occurred', 'Area ID', 'Area Name', 
               'Crime Code', 'Crime Code Description', 'Victim Sex','Victim Descent', 
               'Premise Code', 'Premise Description',
               'Weapon Used Code', 'Weapon Description', 
               'Status Code','Status Description', 
               'Crime Code 1', 'Crime Code 2', 'Crime Code 3','Crime Code 4', 'Location ']]

    #crime code into 4 category (dummies)
    crime['Crime Code 1'] = crime['Crime Code 1'].apply(lambda x: 0 if pd.isna(x) else 1)
    crime['Crime Code 2'] = crime['Crime Code 2'].apply(lambda x: 0 if pd.isna(x) else 1)
    crime['Crime Code 3'] = crime['Crime Code 3'].apply(lambda x: 0 if pd.isna(x) else 1)
    crime['Crime Code 4'] = crime['Crime Code 4'].apply(lambda x: 0 if pd.isna(x) else 1)

    crime = crime.dropna(axis=0)

    #date (month, date), and time (hour,  minute)
    day = crime['Date Occurred'].apply(lambda x: int(str(x)[8:10]))
    month = crime['Date Occurred'].apply(lambda x: int(str(x)[5:7]))
    hour = crime['Time Occurred'].apply(lambda x: int(str(int(x))[:-2]) if str(int(x))[:-2] != '' else 0)
    minute = crime['Time Occurred'].apply(lambda x: int(str(int(x))[-2:]) if str(int(x))[-2:] != '' else 0)

    #location (Geographic Areas, latitude, longitude)
    dummies_area_ID = pd.get_dummies(crime['Area ID']) 
    latitude = crime['Location '].apply(lambda x: float(ast.literal_eval(x)['latitude']))
    longitude = crime['Location '].apply(lambda x: float(ast.literal_eval(x)['longitude']))
    #dummies_type_location = pd.get_dummies(crime['Premise Code']) 

    need_normalize = [day, month, hour, minute, latitude, longitude]
    for i in range(len(need_normalize)):
        need_normalize[i] = normalize(need_normalize[i])


    #crime code in four category (1 most serious)
    code1 = crime['Crime Code 1']
    code2 = crime['Crime Code 2']
    code3 = crime['Crime Code 3']
    code4 = crime['Crime Code 4']


    df_feature_crime = pd.concat({'day':need_normalize[0], 'month':need_normalize[1], 
                            'hour':need_normalize[2], 'minute': need_normalize[3], 
                            'lat':need_normalize[4], 'lon':need_normalize[5]
                            },axis=1)

    #Victim
    victim_sex = pd.get_dummies(crime['Victim Sex'])
    victim_race = pd.get_dummies(crime['Victim Descent'])

    # status of the case
    status = pd.get_dummies(crime['Status Code'])

    merge_dummies_col = [dummies_area_ID, victim_sex, victim_race, status, code1, code2, code3, code4]
    for col in merge_dummies_col:
        col = col.reset_index()
        df_feature_crime = df_feature_crime.merge(col, left_index=True, right_index=True)

    df_feature_crime = df_feature_crime.drop(columns = ['index_x', 'index_y'])

    return df_feature_crime







