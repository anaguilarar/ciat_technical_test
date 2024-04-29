
import argparse
import logging
import numpy as np
import os
import pandas as pd
import pickle
import yaml

from models.utils import setup_model, compute_loss
from models.datasets import DataSplit
from typing import Tuple, Dict
from tqdm import tqdm

from sklearn.base import BaseEstimator


def setup_configuration(config_path):
    if not os.path.exists(config_path):
        raise ValueError(f"{config_path} path does not exists")
    
    with open(config_path, 'r') as fn:
        opt =  yaml.load(fn, Loader=yaml.SafeLoader)
    
    return opt

def train_one_kfold(x_tr: np.ndarray ,
                    y_tr:  np.ndarray,
                    x_eval: np.ndarray,
                    y_eval: np.ndarray,
                    model: BaseEstimator):
    
    #fit the model
    model.fit(x_tr, y_tr)

    #predict
    y_predicted = model.predict(x_eval)

    # evaluate the model
    eval_metrics = compute_loss(y_obs=y_eval, y_pred=y_predicted)

    return model, eval_metrics


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description='Run random regression model for computing mean decrease in impurity.')
    parser.add_argument('--config', 
                        default='configuration.yaml', help='config file path')

    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    # upload configuration
    config = setup_configuration(args.config)

    ## setup data
    data_for_training = DataSplit(path=config['DATASET']['path'], nkfolds=config['DATASPLIT']['kfolds'])

    # check ouput path
    if not os.path.exists(config['MODEL']['output']): os.makedirs(config['MODEL']['output'])

    importnacedf = []
    # iterate k folds
    logging.info(f"""Starting cross-validation step
                 model {config['MODEL']['model_name']}
                 k-folds {config['DATASPLIT']['kfolds']}""")

    for kfold in tqdm(range(0,config['DATASPLIT']['kfolds'])):
        
        # setup model
        model = setup_model(model_name=config['MODEL']['model_name'], cv = config['MODEL']['cv'], 
                        param_grid= config['MODEL']['grid_search_params'])
        
        # 
        training_data, validation_data = data_for_training.kfold_data(kfold)
        
        model, eval_metrics = train_one_kfold(x_tr = training_data[config['DATASET']['features']].values,
                                              y_tr = training_data[config['DATASET']['target']].values,
                                              x_eval = validation_data[config['DATASET']['features']].values,
                                              y_eval = validation_data[config['DATASET']['target']].values,
                                              model = model)

        print(f'KFold: {kfold} Evaluation metrics: {eval_metrics}')

        # save the model if it is needed
        if config['MODEL']['save']:
            fn = '{}_k_{}_of_{}.pickle'.format(
                config['MODEL']['model_name'],
                kfold, config['MODEL']['cv'])
            with open(os.path.join(config['MODEL']['output'], fn), "wb") as file:
                pickle.dump(model, file)

        importnacedf.append(
            pd.DataFrame({
            'kfold': kfold,
            'r2': eval_metrics['r2'],
            'rmse': eval_metrics['rmse'],
            'features':config['DATASET']['features'],
            'mdi': model['rf'].best_estimator_.feature_importances_}))

    outputpath = os.path.join(config['MODEL']['output'],'mdi_kfolds{}.csv'.format(config['DATASPLIT']['kfolds']))
    
    logging.info(f"""Ending the process
                 results saved in {outputpath}""")
    
    pd.concat(importnacedf).to_csv(
        outputpath
    )


if __name__ == "__main__":
    main()