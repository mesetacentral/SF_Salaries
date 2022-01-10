from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, ARDRegression
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
import numpy as np

import os
from joblib import dump, load

import warnings
warnings.filterwarnings('ignore') 

pipe = Pipeline([
    ('model',Ridge())
])

ridge_search = {
    'model': [Ridge()],
    'model__alpha': (1e-8, 1e+2, 'log-uniform'),
}

lasso_search = {
    'model': [Lasso(max_iter=10000)],
    'model__alpha': (1e-8, 1e+2, 'log-uniform'),
}

elasticnet_search = {
    'model': [ElasticNet()],
    'model__alpha': (1e-8, 1e+2, 'log-uniform'),
}

bayesianridge_search = {
    'model': [BayesianRidge()],
    'model__alpha1': (1e-8, 1e+2, 'log-uniform'),
    'model__alpha2': (1e-8, 1e+2, 'log-uniform'),
    'model__lambda1': (1e-8, 1e+2, 'log-uniform'),
    'model__lambda2': (1e-8, 1e+2, 'log-uniform'),
}

ARD_search = {
    'model': [ARDRegression()],
    'model__alpha1': (1e-8, 1e+2, 'log-uniform'),
    'model__alpha2': (1e-8, 1e+2, 'log-uniform'),
    'model__lambda1': (1e-8, 1e+2, 'log-uniform'),
    'model__lambda2': (1e-8, 1e+2, 'log-uniform'),
}

sgd_search = {
    'model': [SGDRegressor(max_iter=500)],
    'model__alpha': (1e-4, 1e+4, 'log-uniform'),
    'model__loss': ['huber', 'squared_error', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
    'model__penalty': ['l1', 'l2', 'elasticnet'],
}

pa_search = {
    'model': [PassiveAggressiveRegressor()],
    'model__C': (1e-2, 1e+2, 'log-uniform'),
}

huber_search = {
    'model': [HuberRegressor()],
    'model__alpha': (1e-4, 1e+4, 'log-uniform'),
    'model__epsilon': (1e+0, 1e+2, 'log-uniform'),
}

rf_search = {
    'model': [RandomForestRegressor()],
    'model__criterion': ['squared_error', 'absolute_error'],
    'model__max_leaf_nodes': (20, 50),
    'model__max_depth': (5, 15),
}

adaboost_search = {
    'model': [AdaBoostRegressor()],
    'model__n_estimators': (20, 100),
    'model__learning_rate': (1e-3, 1, 'log-uniform'),
    'model__loss': ['linear', 'square', 'exponential']
}

bagging_search = {
    'model': [BaggingRegressor()],
    'model__n_estimators': (5, 20),
    'model__max_samples': (1e-2, 1e+0, 'log-uniform')
}

extratrees_search = {
    'model': [ExtraTreesRegressor()],
    'model__n_estimators': (20, 200),
    'model__max_leaf_nodes': (20, 50),
    'model__max_depth': (5, 15),
}

gb_search = {
    'model': [GradientBoostingRegressor()],
    'model__n_estimators': (20, 200),
    'model__learning_rate': (1e-3, 1, 'log-uniform'),
    'model__loss': ['huber', 'squared_error', 'absolute_error', 'quantile']
}

search_grids = [(ridge_search, 10), (lasso_search, 10), (elasticnet_search, 10), (bayesianridge_search, 10), (ARD_search, 10), (sgd_search, 10), (pa_search, 10), (huber_search, 10), (rf_search, 10), (adaboost_search, 10), (bagging_search, 10), (extratrees_search, 10), (gb_search, 10)]

def train_model(demo):
    X_train = np.load('test/X_train.npy')
    X_test = np.load('test/X_test.npy')                                  
    y_train = np.load('test/y_train.npy')    
    y_test = np.load('test/y_test.npy')
    print('Data loaded')
    
    if demo == 'demo':
        models = [Ridge(), PassiveAggressiveRegressor(), ElasticNet()]
        print('Using demo models')
    else:
        models = [Ridge(), Lasso(max_iter=10000), ElasticNet(), BayesianRidge(), ARDRegression(), SGDRegressor(), PassiveAggressiveRegressor(), HuberRegressor(), RandomForestRegressor(), AdaBoostRegressor(), BaggingRegressor(), ExtraTreesRegressor(), GradientBoostingRegressor()]
        print('Using all models')

    
    print('Model fitting')
    scores = list()
    for model in models:
        model.fit(X_train, y_train)
        scores.append((str(model), model.score(X_test, y_test)))
    
    scores = np.array(scores)
    best_scores = scores[:, 1].argsort()[::-1]
   
    final_grid = list()
    for i in best_scores[:3]:
        final_grid.append(search_grids[i])

    opt = BayesSearchCV(
        pipe,
        final_grid,
        cv=2
    )
    print('Hyperparameter tuning')
    opt.fit(X_train, y_train)
    best_model = opt.best_estimator_
    
    dump(best_model, 'models/' + type(best_model).__name__)
    
    return 0

var = input()
train_model(var)
    
        
    


