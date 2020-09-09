import math
import numpy as np
import pandas as pd  
from sklearn import preprocessing

columns = [
			"close",
			"volume",
			"ewm8","ewm13","ewm21","ewm55","StotchRSI"
			
		]

def normalise_windows2(window_data):
    std_scale = preprocessing.StandardScaler().fit(window_data)
    x_train_norm = std_scale.transform(window_data)
    print(x_train_norm)
    return x_train_norm
def normalise_windows(self, window_data, single_window):
    '''Normalise window with a base value of zero'''
    normalised_data = []
    counter = 0
    window_data = [window_data] if single_window else window_data
    for window in window_data:
        normalised_window = []
        for col_i in range(window.shape[1]):
            normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
            normalised_window.append(normalised_col)
        normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
        normalised_data.append(normalised_window)
    return np.array(normalised_data)


data = pd.read_csv("data/2hdata.csv",sep=',')
data = data.get(columns)
# data = np.array(data).astype(float)
# data=((data-data.min())/(data.max()-data.min()))
data = normalise_windows2(data)
print(data)

np.savetxt("test.csv", data)

