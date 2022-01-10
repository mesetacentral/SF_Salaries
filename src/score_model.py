from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import os
from joblib import dump, load
import numpy as np


def score_model(model_name='Pipeline'):
    path = 'models/'
    model = load(os.path.join(path, model_name))
    print('Model loaded')
    
    path = 'test/'
    X_train = np.load(os.path.join(path, 'X_train.npy'))
    X_test = np.load(os.path.join(path, 'X_test.npy'))                                  
    y_train = np.load(os.path.join(path, 'y_train.npy'))    
    y_test = np.load(os.path.join(path, 'y_test.npy'))
    print('Data loaded')
    
    print("R2 score:  ", r2_score(y_test, model.predict(X_test)))    
    print("MSE score: ", mean_squared_error(y_test, model.predict(X_test)))
    print("MAE score: ", mean_absolute_error(y_test, model.predict(X_test)))

score_model()
    