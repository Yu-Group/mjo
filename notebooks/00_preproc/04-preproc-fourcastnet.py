import os
import pathlib

from mjonet import era5

import dask.config
from dask.distributed import Client, LocalCluster


if __name__ == "__main__":

    USER = os.environ['USER']

    if os.environ['NERSC_HOST'] == 'perlmutter':
        SCRATCH = f'/pscratch/sd/{USER[0]}/{USER}/'
    else:
        SCRATCH = f'/global/cscratch1/sd/{USER}/'

    save_dir = os.path.join(SCRATCH, 'data/era5/fourcastnet')

    if not os.path.exists(save_dir):
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    n_workers = os.environ['SLURM_CPUS_ON_NODE']
    chunksize = 40
    n_cached = 280 # ~7 tars per compute
    samples_per_tar = 28 # one week, about 4.4GB per file

    print(f'n_workers: {n_workers}')
    print(f'chunksize: {chunksize}')
    print(f'n_cached: {n_cached}')
    print(f'samples_per_tar: {samples_per_tar}\n')


    with LocalCluster(n_workers=n_workers) as cluster, Client(cluster) as client:
        with dask.config.set({'temporary-directory': '/tmp'}):

            stages = [('validate', range(2016, 2018)),
                      ('train', range(1979, 2016)),
                      ('test', range(2018, 2022))]

            for stage, years in stages:

                glob_dict = era5.make_glob_dict(exclude_vars=['mtnlwrf'], years=years)
                fpath_dict = era5.get_filepaths(glob_dict)

                era5.create_wds(fpath_dict,
                                save_dir=save_dir,
                                samples_per_tar=samples_per_tar,
                                stage=stage,
                                target_steps=[6],
                                plevels={
                                    'z': [50, 500, 850, 1000],
                                    't': [500, 850],
                                    'u': [500, 850, 1000],
                                    'v': [500, 850, 1000],
                                    'r': [500, 850],
                                },
                                chunks={ 'time': chunksize },
                                n_cached=n_cached,
                                parallel=True,
                                verbose=True)
