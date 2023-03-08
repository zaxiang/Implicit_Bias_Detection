from KMeans import KMeans_custom
import numpy as np

class KMenas_custom_runner():
    # This function initializes the KMeans runner class
    def __init__(self, label_series, df_feature, num_cluster, num_iter, order):

        self.X = df_feature.values
        self.y = pd.factorize(label_series)[0]
        
        self.KMeans_custom_model = KMeans_custom(num_cluster, num_iter, order).fit(self.X)
        

    def get_results_df(self):
        race_code_dic = {0:'O', 1:'H', 2:'B', 3:'W', 4:'A', 5:'G', 6:'F', 7:'J', 8:'I', 9:'C', 10:'K', 11:'P', 12:'X', 13:'U', 14:'Z', 15:'S', 16:'L'}

        y = self.y.tolist()
        self.y_pred = self.KMeans_custom_model.predict(self.X)

        self.df_results_custom = pd.concat({'race_label':pd.Series(y).map(race_code_dic), 
                                       'grouping':pd.Series(self.y_pred)}, axis=1) 

        return self.df_results_custom

    def get_center(self):
        self.center = self.KMeans_custom_model.get_centers()
        return self.center

    def get_loss(self):
        self.loss = self.KMeans_custom_model.loss(self.X)
        
        return self.loss
