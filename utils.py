import pandas as pd
import pydicom
# import nvjpeg2k
import numpy as np
from timeit import default_timer as timer
from types import SimpleNamespace
import yaml


def load_config(path: str):
    with open(path, 'r') as fr:
        cfg = yaml.safe_load(fr)
        for k, v in cfg.items():
            if type(v) == dict:
                cfg[k] = SimpleNamespace(**v)
        cfg = SimpleNamespace(**cfg)
    return cfg


def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)

    else:
        raise NotImplementedError
        
        
def make_transfer_syntax_uid(df, dcm_dir):
    machine_id_to_transfer = {}
    machine_id = df.machine_id.unique()
    for i in machine_id:
        d = df[df.machine_id == i].iloc[0]
        f = f'{dcm_dir}/{d.patient_id}/{d.image_id}.dcm'
        dicom = pydicom.dcmread(f)
        machine_id_to_transfer[i] = dicom.file_meta.TransferSyntaxUID
    return machine_id_to_transfer
