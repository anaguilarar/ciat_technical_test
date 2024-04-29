import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold
from typing import List, Optional

def split_idsintwo(data_length: int, ids: List[int] = None, percentage:float = None, seed = 123):
    """
    Split the IDs into two sets.

    Args:
        data_length (int): Length of the IDs.
        ids (list): List of IDs.
        percentage (float): Percentage of data to allocate into one group.
        seed (int): Random seed.

    Returns:
        tuple: A tuple containing both groups of ids.
    """
    if ids is None:
        ids = list(range(len(data_length)))

    if percentage is not None:
        idsremaining = pd.Series(ids).sample(int(data_length*percentage), random_state= seed).tolist()
        main_ids = [i for i in ids if i not in idsremaining]

    else:
        idsremaining = None
        main_ids = ids

    return main_ids, idsremaining

def split_dataintotwo(data, idsfirst, idssecond):
    subset1 = data.iloc[idsfirst]
    subset2 = data.iloc[idssecond]

    return subset1, subset2

def retrieve_datawithids(data, ids):
    if len(ids) > 0:
        subset  = data.iloc[ids]
    else:
        subset = None

    return subset

class SplitIds(object):

    """
    A class to manage splitting of indices for dataset into training and testing sets,
    with optional support for k-fold cross-validation.

    Parameters
    ----------
    data_length : int, optional
        Length of the data for which indices are required.
    ids : List[int], optional
        A predefined list of indices.
    nkfolds : int, optional
        Number of folds for k-fold cross-validation.
    test_perc : float, optional
        Percentage of data to be reserved for testing.
    seed : int, optional
        Random seed for reproducibility.
    shuffle : bool, optional
        Whether to shuffle the indices.

    Attributes
    ----------
    training_ids : list
        List of indices for training data.
    test_ids : list
        List of indices for testing data, if test_perc is provided.

    Raises
    ------
    ValueError
        If both data_length and ids are None or if both are provided.
    """

    def _ids(self):
        """
        Generate a list of IDs ranging from 0 to (ids_length - 1).

        Returns:
            list: A list of IDs.
        """
        
        ids = list(range(self.data_length))
        if self.shuffle:
            ids = pd.Series(ids).sample(n = self.data_length, random_state= self.seed).tolist()

        return ids


    def kfolds(self, kfolds, shuffle = True):
        """
        Generate k-fold datasets for training and validation.

        Parameters
        ----------
        kfolds : int
            Number of folds.
        shuffle : bool, optional
            Whether to shuffle indices before splitting.

        Returns
        -------
        list of lists
            A list containing train-test indices pairs for each fold.
        """

        kf = KFold(n_splits=kfolds, shuffle = shuffle, random_state = self.seed)

        idsperfold = []
        for train, test in kf.split(self.training_ids):
            idsperfold.append([list(np.array(self.training_ids)[train]),
                               list(np.array(self.training_ids)[test])])

        return idsperfold
    
    def __init__(self, data_length: Optional[int] = None, ids: List[int] = None, nkfolds:int = None, test_perc:float = None,seed:int = 123,
                  shuffle: bool = True) -> None:
        
        
        self.shuffle = shuffle
        self.seed = seed
        
        if ids is None and data_length is not None:
            self.data_length = data_length
            self.ids = self._ids()
        elif data_length is None and ids is not None:
            self.data_length = len(ids)
            self.ids = ids
        else:
            raise ValueError ("provide an index list or a data length value")
        
        self.nkfolds = nkfolds
        
        if test_perc is not None:
            self.training_ids, self.test_ids = split_idsintwo(self.data_length, self.ids, test_perc, self.seed)
        else:
            self.training_ids = range(self.data_length)



class DataReader():
    """
    A class to read data from a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Raises
    ------
    ValueError
        If the provided path does not exist.
    """

    def __init__(self, path:str) -> None:
        self.file_path = self._check_file_path(path)
        self._df = None
    
    @staticmethod
    def _check_file_path( path):
        if not os.path.exists(path):
            raise ValueError(f"The path {path} does not exists")
        return path
    
    @property
    def data(self) -> pd.DataFrame:
        """
        Load data from the CSV file if not already loaded.

        Returns
        -------
        DataFrame
            Data loaded from the CSV file.
        """

        if self._df is None:
            self._df = pd.read_csv(self.file_path) 
        
        return self._df



class DataSplit(DataReader, SplitIds):

    
    @property
    def test_data(self):
        """Retrieves the test subset of the data."""
        return retrieve_datawithids(self.data, self.test_ids) 
    
    @property
    def training_data(self):
        """Retrieves the training subset of the data."""
        return retrieve_datawithids(self.data, self.training_ids) 
    

    def kfold_data(self, kifold):
        """
        Returns training and validation data subsets for the specified K-fold index.

        Parameters
        ----------
        kifold : int
            The index of the fold for which to retrieve the data.

        Returns
        -------
        Tuple[Optional[Any], Optional[Any]]
            A tuple containing the training and validation datasets for the specified fold.
            Returns (None, None) if K-folds are not defined or if the fold index is out of range.
        """
        tr, val = None, None
        if self.nkfolds is not None:
            if kifold <= self.nkfolds:
                tr, val = split_dataintotwo(self.data, 
                                            idsfirst = self.kfolds(self.nkfolds)[kifold][0], 
                                            idssecond = self.kfolds(self.nkfolds)[kifold][1])

        return tr, val
        
    def __init__(self, path, nkfolds:int = None, test_perc:float = None, seed:int = 123,
                  shuffle: bool = True) -> None:
        """
        Initializes the SplitData instance with the dataset, an object for ID partitions, and an optional K-fold count.

        Parameters
        ----------
        df : Any
            The complete dataset.
        split_ids : Any
            An object containing ID partitions for splitting the data.
        kfolds : Optional[int], optional
            The number of folds for K-fold cross-validation, by default None.
        """
        DataReader.__init__(self, path)
        SplitIds.__init__(self, self.data.shape[0], nkfolds=nkfolds, test_perc=test_perc, seed=seed, shuffle=shuffle)



