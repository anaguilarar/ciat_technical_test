import numpy as np

from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.fixes import loguniform
from sklearn.svm import SVR

from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,  MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

def compute_loss(y_obs, y_pred):
    """
    Compute r square and root meand squared error for the model predictions against true values.

    Parameters
    ----------
    y_pred : ndarray
        Predictions made by the model.
    y_obs : ndarray
        True values.

    Returns
    -------
    dict
        A dictionary containing computed metrics.
    """
    model_accuracy = np.sqrt(mean_squared_error(y_true=y_obs, y_pred=y_pred))
    model_f1score = r2_score(y_true=y_obs, y_pred=y_pred)
    
    losses =  {'r2':model_f1score,'rmse':model_accuracy}

    return losses


def setup_model(model_name = 'pls',
              scaler = 'standardscaler', 
              param_grid = None, 
              scale_data = True,
              cv = 5, 
              nworkers = -1):
    
    """
    function to set a shallow learning model for regression, this is a sklearn function which first will scale the data, then will 
    do a gridsearch to find the best hyperparameters

    Parameters:
    ----------
    model_name: str
        which is the model that will be used
        {'pls': Partial least square,
         'svr_radial': support vector machine with radial kernel,
         'svr_linear': support vector machine with linear kernel,
         'rf': Random Forest,
         'lasso', 'ridge', 'default': 'pls'}
    scaler: str
        which data scaler will be applied
        {'minmax', 'standardscaler', default: 'standardscaler'}
    param_grid: dict, optional
        grid parameters used for hyperparameters gird searching
        
    scale_data: boolean, optional
        use scaler in the model
    cv: int
        k-folds for cross-validation
    nworkers: int
        set the number of workers that will be used for parallel process

    Returns:
    ---------
    pipemodel

    """
    if scaler == 'minmax':
        scl = MinMaxScaler()
    if scaler == 'standardscaler':
        scl = StandardScaler()

    if model_name == 'pls':
        if param_grid is None:
            rdcomps = np.linspace(start = 1, stop = 50, num = 30)
            param_grid = [{'n_components':np.unique([int(i) for i in rdcomps])}]

        mlmodel = GridSearchCV( PLSRegression(),
                                param_grid,
                                cv=cv,
                                n_jobs=nworkers)

    if model_name == 'svr_linear':
        if param_grid is None:
            param_grid = {'C': loguniform.rvs(0.1, 1e3, size=20),
                          'gamma': loguniform.rvs(0.0001, 1e-1, size=20)}

        ## model parameters
        mlmodel  = GridSearchCV(SVR(kernel='linear'),
                                          param_grid,
                                          cv=cv,
                                          n_jobs=nworkers)


    if model_name == 'svr_radial':
        if param_grid is None:
            param_grid = {'C': loguniform.rvs(0.1, 1e3, size=20),
                          'gamma': loguniform.rvs(0.0001, 1e-1, size=20)}
        ## model parameters
        mlmodel  = GridSearchCV(SVR(kernel='rbf'),
                                          param_grid,
                                          cv=cv,
                                          n_jobs=nworkers)



    if model_name == 'xgb':
        if param_grid is None:
            param_grid = {
                    'min_child_weight': [1, 2, 4],
                    'gamma': [0.001,0.01,0.5, 1, 1.5, 2, 5],
                    'n_estimators': [100, 500],
                    'colsample_bytree': [0.7, 0.8],
                    'max_depth': [2,4,8,16,32],
                    'reg_alpha': [1.1, 1.2, 1.3],
                    'reg_lambda': [1.1, 1.2, 1.3],
                    'subsample': [0.7, 0.8, 0.9]
                    }

        xgbreg = xgb.XGBRegressor(
                        eval_metric="rmse",
                        random_state = 123
                )
        mlmodel  = RandomizedSearchCV(xgbreg,
                               param_grid,
                               cv=cv,
                               n_jobs=nworkers,
                               n_iter = 50)

       
    if model_name == 'rf':
        if param_grid is None:
            param_grid = { 
            'n_estimators': [200],
            'max_features': [ 0.4, 0.6, 0.8],
            'max_depth' : [2,4,8,16,32],
            'min_samples_split' : [2,4,8],
            'max_samples': [0.7,0.9],
                        
            #'max_leaf_nodes': [50, 100, 200]
            #'criterion' :['gini', 'entropy']
            }
        mlmodel = GridSearchCV( RandomForestRegressor(random_state = 42),
                                param_grid,
                                cv=5,
                                n_jobs=-1)
    

    
    if model_name == 'lasso':
        if param_grid is None:
            alphas = np.logspace(-4, -0.5, 30)
            param_grid = [{"alpha": alphas}]
            
        mlmodel  = GridSearchCV(Lasso(random_state=0, max_iter=4000),
                                param_grid,
                                cv=cv,
                                n_jobs=nworkers)

    if model_name == 'ridge':
        if param_grid is None:
            alphas = np.logspace(-4, -0.5, 30)
            param_grid = [{"alpha": alphas}]
            
        mlmodel  = GridSearchCV(Ridge(random_state=0, max_iter=4000),
                                param_grid,
                                cv=cv,
                                n_jobs=nworkers)
        
    
    if scale_data:
        pipelinemodel = Pipeline([('scaler', scl), (model_name, mlmodel)])
    else:
        pipelinemodel = Pipeline([(model_name, mlmodel)])


    return pipelinemodel
