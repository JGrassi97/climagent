import os, shutil
import tempfile
import requests
import pandas as pd
import xarray as xr
import numpy as np
from tqdm import tqdm
import s3fs



def _get_url(collection : str, variable : str, dataset : str, scenario : str, product : str, aggregation : str, statistic : str, type : str, percentile : str, period : str) -> str:
    """
    Build the url to request the file from the s3 bucket.
    """
    base_url = 'https://wbg-cckp.s3.amazonaws.com/data'
    url = f'{base_url}/{collection}/{variable}/{dataset}-{scenario}/{product}-{variable}-{aggregation}-{statistic}_{collection}_{dataset}-{scenario}_{type}_{percentile}_{period}.nc'

    return url



def _download_file(collection : str, variable : str, dataset : str, scenario : str, product : str, aggregation : str, statistic : str, type : str, percentile : str, period : str, dest_folder : str):

    url = _get_url(collection, variable, dataset, scenario, product, aggregation, statistic, type, percentile, period)
    local_filename = os.path.join(dest_folder, os.path.basename(url))

    try:

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(local_filename, 'wb') as f, tqdm(
                desc=local_filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:  
                        f.write(chunk)
                        bar.update(len(chunk))
        
        return 0
    
    except:
        return 1


def _subset_dataset(dataset : xr.Dataset, lat : list, lon : list) -> xr.Dataset:
    """
    Subset the dataset to a specific lat and lon.
    """
    if lat is not None and lon is not None:
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


def _rename_variable(dataset : xr.Dataset, new_name : str) -> xr.Dataset:
    """
    Rename the variable in the dataset.
    """

    var_name = list(dataset.data_vars)[0]
    dataset = dataset.rename_vars({var_name: new_name})

    return dataset
    

def _save_dataset(dataset: xr.Dataset, directory : str, name : str) -> None:
    """
    Save the dataset to csv
    """
    var = list(dataset.data_vars)[0]

    dataset[var].encoding['_FillValue'] = np.nan
    dataset[var].encoding['missing_value'] = np.nan

    dataset.to_netcdf(f'{directory}/{name}.nc', )



def _convert_timedelta_to_days(timedelta_str):
    
    return float(pd.Timedelta(timedelta_str).total_seconds() / 86400)


def _convert_days(dataset : xr.Dataset) -> xr.Dataset:

    var_name = list(dataset.data_vars)[0]
    dataset[var_name] = xr.apply_ufunc(np.vectorize(_convert_timedelta_to_days), dataset[var_name])

    return dataset



def retrieve_dataset(collection, variable, dataset, scenario, product, aggregation, statistic, type, percentile, period,
                     lat, lon, forbidden_list, cd, directory, name):
    """
    Retrieve the dataset from the s3 bucket, subset it and save it to csv.
    """

    stat = _download_file(collection, variable, dataset, scenario, product, aggregation, statistic, type, percentile, period, directory)

    if stat == 1:
        return 1

    file_name = f'{directory}/{os.path.basename(_get_url(collection, variable, dataset, scenario, product, aggregation, statistic, type, percentile, period))}'
    
    with xr.open_dataset(file_name) as dataset:

        dataset = _subset_dataset(dataset, lat, lon)
        dataset = _clean_dataset(dataset, forbidden_list)
        dataset = _rename_variable(dataset, variable)

        if cd:
            dataset = _convert_days(dataset)

        dataset['time'] = pd.to_datetime(dataset.time.values)

        _save_dataset(dataset, directory, name)

    os.remove(file_name)
