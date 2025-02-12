import requests
import os, shutil
import xarray as xr
from tqdm import tqdm
import pandas as pd
import argparse
import warnings
import multiprocessing
import zarr

from climagent.cckp import _download_file, retrieve_dataset


def set_enviroment(path : str):

    temp_dir = f'{path}/tmp'
    store_dir = f'{path}/store'

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    if not os.path.exists(store_dir):
        os.makedirs(store_dir)

    return temp_dir, store_dir


# 
def task(args):

    collection, variable, dataset, scenario, product, aggregation, statistic, type, percentile, period, lat, lon, forbidden_list, convert_days, directory  = args

    retrieve_dataset(collection, variable, dataset, scenario, product, aggregation, statistic, 
                     type, percentile, period, lat, lon, 
                     forbidden_list, convert_days, directory, f'{variable}_{period}_{scenario}_{percentile}')
    


def download_temp_data(collection, variables, dataset, scenarios, product, aggregation, statistics, type, percentiles, periods, lat, lon, forbidden_list, convert_days, directory, processes):
    """
    Download the files from the s3 bucket using multiprocessing.
    """

    tasks = [(collection, variable, dataset, scenario, product, aggregation, statistic, type, percentile, period, lat, lon, forbidden_list, convert_day, directory) 
             for variable, statistic, convert_day in zip(variables, statistics, convert_days) 
             for percentile in percentiles
             for scenario in scenarios
             for period in periods]

    with multiprocessing.Pool(processes=processes) as pool:
        pool.map(task, tasks) 
            

def create_zarr_archive(variables, periods, percentiles, scenarios, product, temp_dir, store_dir):
    """
    Create a zarr archive from the downloaded files.
    """

    store_path = f'{store_dir}/{product}_timeseries.zarr'
    
    # Se il dataset esiste già, rimuovilo per evitare errori
    if os.path.exists(store_path):
        import shutil
        shutil.rmtree(store_path)

    # first = True  # Flag per la prima scrittura

    # for var in variables:
    #     for period in periods:
    #         for percentile in percentiles:
    #             for scenario in scenarios:
    #                 try:
    #                     single_dat = xr.open_dataset(f'{temp_dir}/{var}_{period}_{scenario}_{percentile}.nc')
                        
    #                     # Convertire stringhe in object o rimuoverle
    #                     for var_name, da in single_dat.data_vars.items():
    #                         if da.dtype.kind in {'U', 'S'}:  # Stringhe
    #                             single_dat[var_name] = da.astype(object)

    #                     # Aggiungi dimensione scenario
    #                     single_dat = single_dat.expand_dims('scenario')
    #                     single_dat['scenario'] = [scenario]

    #                     # Aggiungi dimensione percentile
    #                     single_dat = single_dat.expand_dims('percentile')
    #                     single_dat['percentile'] = [percentile]

    #                     # Scrivi direttamente su Zarr in modo incrementale
    #                     if first:
    #                         single_dat.to_zarr(store_path, mode='w')
    #                     else:
    #                         single_dat.to_zarr(store_path, mode='a', append_dim='scenario')
                        
    #                     first = False  # Dopo il primo salvataggio, passa a modalità append

    #                 except Exception as e:
    #                     print(f"Errore nel file {var}_{period}_{scenario}_{percentile}.nc: {e}")
        
    dats = []
    for var in variables:
        for period in periods:
            for percentile in percentiles:
                for scenario in scenarios:

                    try:
                        single_dat = xr.open_dataset(f'{temp_dir}/{var}_{period}_{scenario}_{percentile}.nc')

                        # Add dimension for scenario
                        single_dat = single_dat.expand_dims('scenario')
                        single_dat['scenario'] = [scenario]

                        # Add dimension for percentile
                        single_dat = single_dat.expand_dims('percentile')
                        single_dat['percentile'] = [percentile]
                        
                        dats.append(single_dat)

                    except:
                        pass

    dats = xr.merge(dats)
    #dats = dats.chunk({'time': -1, 'percentile': -1, 'scenario': -1, 'lat': 181, 'lon': 720})
    dats.to_zarr(store_path, mode='w')





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

    parser.add_argument(
        "--sources",
        help="Sources to be downloaded",
        type=str,
        default=['ERA5', 'CMIP6'],
        required=False
    )

    parser.add_argument(
        "--path",
        help="Path for the store",
        type=str,
        default='example',
        required=False
    )


    processes = parser.parse_args().processes
    sources = parser.parse_args().sources
    path = parser.parse_args().path

    temp_dir, store_dir = set_enviroment(path)
    lat = [25,50]
    lon = [-15,50]
    # lat = None
    # lon = None
    forbidden_list = ['lat_bnds', 'lon_bnds', 'bnds']

    # variables = ['tas', 'tasmin', 'tasmax', 'pr', 'cdd', 'cdd65', 'fd', 'hd30', 'hd35', 'hd40' , 'hd42', 'hd45', 'csdi','hdd65', 'hi', 'hi35', 'hi37', 'hi39', 'hi41', 'r20mm', 'r50mm', 'rx1day', 'rx5day','sd','td','tnn','tr','tr23','tr26','tr29','tr32','txx','wsdi']
    # statistics = ['mean', 'mean', 'mean', 'mean', 'max', 'mean', 'mean','mean','mean','mean','mean','mean','mean','mean','mean','mean','mean','mean','mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'mean']
    # convert_days = [False, False, False, False, True, False, True, True, True, True, True, True, True, False, False, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True,True, True, False, True]

    variables = ['tas', 'tasmin', 'tasmax', 'hd30', 'hd35', 'hd40', 'hi35', 'hi37', 'hi39', 'hi41', 'tr23', 'tr26', 'tr29', 'tr32', 'txx', 'wsdi']
    statistics = ['mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'mean']
    convert_days = [False, False, False, True, True, True, True, True, True, True, True, True, True, True, False, True]
    
    if 'ERA5' in sources:
        # ERA5 ARCHIVE
        
        collection = 'era5-x0.25'
        periods = ['1950-2023']
        dataset = 'era5-x0.25'
        scenarios = ['historical']
        product = 'timeseries'
        aggregation = 'annual'
        type = 'timeseries'
        percentiles = ['mean']
        print('ERA5')
        download_temp_data(collection, variables, dataset, scenarios, product, aggregation, statistics, type, percentiles, periods, lat, lon, forbidden_list, convert_days, temp_dir, processes)
        create_zarr_archive(variables, periods, percentiles, scenarios, 'ERA5', temp_dir, store_dir)
        shutil.rmtree(temp_dir)

    if 'CMIP6' in sources:

        # CMIP6 ARCHIVE
        temp_dir, store_dir = set_enviroment(path)

        collection = 'cmip6-x0.25'
        product = 'timeseries'
        aggregation = 'annual'
        type = 'timeseries'
        percentiles = ['p10','median','p90']
        dataset = 'ensemble-all'

        periods = ['2015-2100']
        scenarios = ['ssp126', 'ssp245','ssp585']

        download_temp_data(collection, variables, dataset, scenarios, product, aggregation, statistics, type, percentiles, periods, lat, lon, forbidden_list, convert_days, temp_dir, processes)

        periods = ['1950-2014']
        scenarios = ['historical']

        download_temp_data(collection, variables, dataset, scenarios, product, aggregation, statistics, type, percentiles, periods, lat, lon, forbidden_list, convert_days, temp_dir, processes)

        periods = ['1950-2014', '2015-2100']
        scenarios = ['historical', 'ssp126', 'ssp245', 'ssp585']

        #create_zarr_archive(variables, periods, percentiles, scenarios, 'CMIP6', temp_dir, store_dir)    
        #shutil.rmtree(temp_dir)
    

