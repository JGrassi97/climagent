import requests
import os, shutil
import xarray as xr
from tqdm import tqdm
import pandas as pd
import argparse
import warnings
import multiprocessing

from climagent.cckp import _download_file, retrieve_dataset


def set_enviroment(path : str):

    temp_dir = f'{path}/tmp'
    store_dir = f'{path}/store'

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    if not os.path.exists(store_dir):
        os.makedirs(store_dir)

    return temp_dir, store_dir


# Funzione globale che sarà eseguita nei processi paralleli
def task(args):
    var, stat, cd, period, temp_dir = args
    retrieve_dataset('era5-x0.25', var, 'era5-x0.25', 'historical', 'timeseries', 'annual', stat, 
                     'timeseries', 'mean', period, None, None, 
                     ['lat_bnds', 'lon_bnds', 'bnds'], cd, temp_dir, f'{var}_{period}')

def download_temp_data(variables, statistics, periods, convert_days, temp_dir, processes):
    """
    Download the files from the s3 bucket using multiprocessing.
    """

    tasks = [(var, stat, cd, period, temp_dir) for var, stat, cd in zip(variables, statistics, convert_days) for period in periods]

    with multiprocessing.Pool(processes=processes) as pool:
        pool.map(task, tasks) 
            

def create_zarr_archive(variables, periods, temp_dir, store_dir):
    """
    Create a zarr archive from the downloaded files.
    """
    
    dats = []
    for var in variables:
        for period in periods:

            try:
                dats.append(xr.open_dataset(f'{temp_dir}/{var}_{period}.nc'))
            except:
                pass


    dats = xr.merge(dats)

    dats.to_zarr(f'{store_dir}/era5-x0.25_timeseries.zarr', mode='w')



def main():

    warnings.filterwarnings("ignore")


    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--processes",
        help="Number of processes for parallel download",
        type=int,
        default=4,
        required=False
    )

    processes = parser.parse_args().processes

    path = '/example'

    variables = ['tas', 'tasmin', 'tasmax', 'pr', 'cdd', 'cdd65', 'fd', 'hd30', 'hd35', 'hd40' , 'hd42', 'hd45', 'csdi','hdd65', 'hi', 'hi35', 'hi37', 'hi39', 'hi41', 'r20mm', 'r50mm', 'rx1day', 'rx5day','sd','td','tnn','tr','tr23','tr26','tr29','tr32','txx','wsdi']
    statistics = ['mean', 'mean', 'mean', 'mean', 'max', 'mean', 'mean','mean','mean','mean','mean','mean','mean','mean','mean','mean','mean','mean','mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'mean']
    convert_days = [False, False, False, False, True, False, True, True, True, True, True, True, True, False, False, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True,True, True, False, True]

    periods = ['1950-2023']

    temp_dir, store_dir = set_enviroment(path)
    download_temp_data(variables, statistics, periods, convert_days, temp_dir, processes)
    create_zarr_archive(variables, periods, temp_dir, store_dir)

    shutil.rmtree(temp_dir)
    

