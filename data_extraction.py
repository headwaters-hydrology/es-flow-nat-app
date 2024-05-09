#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:26:09 2024

@author: mike
"""
from tethysts import Tethys
import booklet
import orjson
import os
import pathlib
import pandas as pd


################################################
### Parameters

script_path = pathlib.Path(os.path.realpath(os.path.dirname(__file__)))

assets_path = script_path.joinpath('app/assets')

remotes = [
  {'bucket': 'es-hilltop',
   'public_url': 'https://b2.tethys-ts.xyz/file',
    'version': 4},
  {'bucket': 'nz-water-use',
   'public_url': 'https://b2.tethys-ts.xyz/file',
    'version': 4}
    ]

ds_ids_dict = {
    'flow_nat': 'dd2a6221a32ba69bc1ee3e4f',
    'rec_flow': '4ae05d099af292fec48792ec',
    'man_flow': '8120cbf5f9e694899fbcb435',
    'use': '7cc8b402e168885ef69870ed',
    'flow_use': 'fde8d7496cb5cda74b8bbe8d',
    'allo': '609ba10d2271c5a18b1c5103',
    'velocity': 'a66a69063e43457e905de6a8',
    'gage_height': '11edd2edbc10c4f51e71cdd6'
    }

ds_ids = list(ds_ids_dict.values())

###############################################
### Extract data

tethys = Tethys(remotes)

## Datasets
dss = [ds for ds_id, ds in tethys._datasets.items() if ds_id in ds_ids]

dss_json = orjson.dumps(dss)

dss_path = assets_path.joinpath('datasets.json')
with open(dss_path, 'wb') as f:
    f.write(dss_json)

## Stations
stns_dict = {}
for ds in dss:
    ds_id = ds['dataset_id']
    stns = tethys.get_stations(ds_id)
    stns_dict[ds_id] = stns


## Results
updated_stns_dict = {}
for ds in dss:
    ds_id = ds['dataset_id']
    print('-- ' + ds_id)
    stns = stns_dict[ds_id]
    stns_len = len(stns)
    print(f'{stns_len} stations')

    ds_path = assets_path.joinpath(ds_id + '.blt')

    updated_stns_list = []
    with booklet.open(ds_path, 'n', key_serializer='str', value_serializer='pickle_zstd') as f:
        perc_init = 0
        for i, stn in enumerate(stns):
            stn_id = stn['station_id']
            try:
                data3 = tethys.get_results(ds_id, stn_id, squeeze_dims=True).load()
            except:
                continue
            data3['time'] = pd.to_datetime(data3['time'].values) + pd.DateOffset(hours=12)
            coords = list(data3.coords)
            if 'geometry' in coords:
                data3 = data3.drop('geometry')
            if 'height' in coords:
                data3 = data3.drop('height')

            f[stn_id] = data3
            updated_stns_list.append(stn)

            perc = round(((i+1)/stns_len) * 100)
            if perc > perc_init:
                print(perc)
                perc_init = perc

    updated_stns_dict[ds_id] = updated_stns_list


## Save stations
stns_path = assets_path.joinpath('stns_data.blt')
with booklet.open(stns_path, 'n', key_serializer='str', value_serializer='orjson_zstd') as f:
    for ds_id, stns in updated_stns_dict.items():
        f[ds_id] = stns






























































