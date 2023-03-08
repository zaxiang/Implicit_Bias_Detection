from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def get_true_and_cluster_value(result_df):
    
    result_df = result_df[result_df['race_label'].isin(['B', 'H', 'W', 'O'])]
    
    dic = {}
    
    dic["true_dist"] = result_df.race_label.value_counts(normalize=True).values
    for clu in result_df['grouping'].unique():
        dic["cluster "+str(clu)] = result_df[result_df['grouping'] == clu].race_label.value_counts(normalize=True).values

    return dic
    
def plot_distribution(result_df):
        
    plotdata = pd.DataFrame(get_true_and_cluster_value(result_df))

    my_colors = ['r', 'g', 'y', 'g', 'y', 'g', 'y'] #for 6 cluster (just to be safe)

    plotdata.plot(kind="bar",figsize=(15, 8), color=my_colors)
    plt.title('Race Distribution: Overall and Within_5_Clusters')
    plt.show()
