import os, shutil
import tempfile

import pandas as pd
import xarray as xr

import s3fs



def _get_url(collection : str, variable : str, dataset : str, scenario : str, product : str, aggregation : str, statistic : str, type : str, percentile : str, period : str) -> str:
    """
    Build the url to request the file from the s3 bucket.
    """

    base_url = 's3://wbg-cckp/data'
    url = f'{base_url}/{collection}/{variable}/{dataset}-{scenario}/{product}-{variable}-{aggregation}-{statistic}_{collection}_{dataset}-{scenario}_{type}_{percentile}_{period}.nc'

    return url


def _request_dataset(collection : str, variable : str, dataset : str, scenario : str, product : str, aggregation : str, statistic : str, type : str, percentile : str, period : str) -> xr.Dataset:
    """
    Request the dataset from the s3 bucket.
    """

    url = _get_url(collection, variable, dataset, scenario, product, aggregation, statistic, type, percentile, period)

    try:
        fs = s3fs.S3FileSystem(anon=True)
        f = fs.open(url)        
        dataset = xr.open_dataset(f, engine='h5netcdf')

        return dataset
    
    except:
        print('File not found')
        return None


def _subset_dataset(dataset : xr.Dataset, lat : list, lon : list) -> xr.Dataset:
    """
    Subset the dataset to a specific lat and lon.
    """

    dataset = dataset.sel(lat=slice(lat[0], lat[1]), lon=slice(lon[0], lon[1]))

    return dataset


def _clean_dataset(dataset : xr.Dataset, forbidden_list : list) -> xr.Dataset:
    """
    Remove the variables in the forbidden list from the dataset.
    """

    for forbidden_word in forbidden_list:
        if forbidden_word in dataset:
            dataset = dataset.drop_vars(forbidden_word)

    return dataset
    

def _save_dataset(dataset: xr.Dataset, directory : str, name : str) -> None:
    """
    Save the dataset to csv
    """

    dataset_df = dataset.to_dataframe()
    dataset_df.to_csv(f'{directory}/{name}.csv', )



def retrieve_dataset(collection, variable, dataset, scenario, product, aggregation, statistic, type, percentile, period,
                     lat, lon, forbidden_list, directory, name):
    """
    Retrieve the dataset from the s3 bucket, subset it and save it to csv.
    """

    dataset = _request_dataset(collection, variable, dataset, scenario, product, aggregation, statistic, type, percentile, period)
    dataset = _subset_dataset(dataset, lat, lon)
    dataset = _clean_dataset(dataset, forbidden_list)
    _save_dataset(dataset, directory, name)