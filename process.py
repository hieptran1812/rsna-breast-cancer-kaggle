"""
this process to convert dicom to png image to load the model
"""
import numpy as np
import pandas as pd
import glob
import cv2
import shutil
import ctypes
from pathlib import Path
from tqdm import tqdm

import pydicom
from pydicom.filebase import DicomBytesIO

import multiprocessing as mp
from joblib import Parallel, delayed

from sklearn.model_selection import StratifiedGroupKFold

from utils import load_config
import torch
import torch.nn.functional as F

from nvidia.dali import pipeline_def, types
from nvidia.dali.types import DALIDataType
from nvidia.dali.backend import TensorGPU, TensorListGPU
import nvidia.dali.fn as fn
import nvidia.dali.types as types



to_torch_type = {
    types.DALIDataType.FLOAT:   torch.float32,
    types.DALIDataType.FLOAT64: torch.float64,
    types.DALIDataType.FLOAT16: torch.float16,
    types.DALIDataType.UINT8:   torch.uint8,
    types.DALIDataType.INT8:    torch.int8,
    types.DALIDataType.UINT16:  torch.int16,
    types.DALIDataType.INT16:   torch.int16,
    types.DALIDataType.INT32:   torch.int32,
    types.DALIDataType.INT64:   torch.int64
}


def feed_ndarray(dali_tensor, arr, cuda_stream=None):
    """
    Copy contents of DALI tensor to PyTorch's Tensor.

    Parameters
    ----------
    `dali_tensor` : nvidia.dali.backend.TensorCPU or nvidia.dali.backend.TensorGPU
                    Tensor from which to copy
    `arr` : torch.Tensor
            Destination of the copy
    `cuda_stream` : torch.cuda.Stream, cudaStream_t or any value that can be cast to cudaStream_t.
                    CUDA stream to be used for the copy
                    (if not provided, an internal user stream will be selected)
                    In most cases, using pytorch's current stream is expected (for example,
                    if we are copying to a tensor allocated with torch.zeros(...))
    """
    dali_type = to_torch_type[dali_tensor.dtype]

    assert dali_type == arr.dtype, ("The element type of DALI Tensor/TensorList"
                                    " doesn't match the element type of the target PyTorch Tensor: "
                                    "{} vs {}".format(dali_type, arr.dtype))
    assert dali_tensor.shape() == list(arr.size()), \
        ("Shapes do not match: DALI tensor has size {0}, but PyTorch Tensor has size {1}".
            format(dali_tensor.shape(), list(arr.size())))
    cuda_stream = types._raw_cuda_stream(cuda_stream)

    # turn raw int to a c void pointer
    c_type_pointer = ctypes.c_void_p(arr.data_ptr())
    if isinstance(dali_tensor, (TensorGPU, TensorListGPU)):
        stream = None if cuda_stream is None else ctypes.c_void_p(cuda_stream)
        dali_tensor.copy_to_external(c_type_pointer, stream, non_blocking=True)
    else:
        dali_tensor.copy_to_external(c_type_pointer)
    return arr


def process(f, save_folder=""):
    patient = f.split('/')[-2]
    dicom_id = f.split('/')[-1][:-4]
    
    dicom = dicomsdl.open(f)
    img = dicom.pixelData()
    img = torch.from_numpy(img)
    img = process_dicom(img, dicom)
    
    img = F.interpolate(img.view(1, 1, img.size(0), img.size(1)), (SAVE_SIZE, SAVE_SIZE), mode="bilinear")[0, 0]

    img = (img * 255).clip(0,255).to(torch.uint8).cpu().numpy()
    out_file_name = SAVE_FOLDER + f"{patient}_{dicom_id}.png"
    cv2.imwrite(out_file_name, img)
    return out_file_name


def convert_dicom_to_jpg(file, save_folder=""):
    patient = file.split('/')[-2]
    image = file.split('/')[-1][:-4]
    dcmfile = pydicom.dcmread(file)

    if dcmfile.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2.4.90':
        with open(file, 'rb') as fp:
            raw = DicomBytesIO(fp.read())
            ds = pydicom.dcmread(raw)
        offset = ds.PixelData.find(b"\x00\x00\x00\x0C")  #<---- the jpeg2000 header info we're looking for
        hackedbitstream = bytearray()
        hackedbitstream.extend(ds.PixelData[offset:])
        with open(save_folder + f"{patient}_{image}.jpg", "wb") as binary_file:
            binary_file.write(hackedbitstream)
            
    if dcmfile.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2.4.70':
        with open(file, 'rb') as fp:
            raw = DicomBytesIO(fp.read())
            ds = pydicom.dcmread(raw)
        offset = ds.PixelData.find(b"\xff\xd8\xff\xe0")  #<---- the jpeg lossless header info we're looking for
        hackedbitstream = bytearray()
        hackedbitstream.extend(ds.PixelData[offset:])
        with open(save_folder + f"{patient}_{image}.jpg", "wb") as binary_file:
            binary_file.write(hackedbitstream)

            
@pipeline_def
def jpg_decode_pipeline(jpgfiles):
    jpegs, _ = fn.readers.file(files=jpgfiles)
    images = fn.experimental.decoders.image(jpegs, device='mixed', output_type=types.ANY_DATA, dtype=DALIDataType.UINT16)
    return images

def parse_window_element(elem):
    if type(elem)==list:
        return float(elem[0])
    if type(elem)==str:
        return float(elem)
    if type(elem)==float:
        return elem
    if type(elem)==pydicom.dataelem.DataElement:
        try:
            return float(elem[0])
        except:
            return float(elem.value)
    return None

def linear_window(data, center, width):
    lower, upper = center - width // 2, center + width // 2
    data = torch.clamp(data, min=lower, max=upper)
    return data 

def process_dicom(img, dicom):
    try:
        invert = getattr(dicom, "PhotometricInterpretation", None) == "MONOCHROME1"
    except:
        invert = False
        
    center = parse_window_element(dicom["WindowCenter"]) 
    width = parse_window_element(dicom["WindowWidth"])
        
    if (center is not None) & (width is not None):
        img = linear_window(img, center, width)

    img = (img - img.min()) / (img.max() - img.min())
    if invert:
        img = 1 - img
    return img


COMP_FOLDER = 'data/'
DATA_FOLDER = 'data/train_images/'

N_CORES = mp.cpu_count()
MIXED_PRECISION = False
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
RAM_CHECK = True
DEBUG = True

train_df = pd.read_csv('data/train.csv')
# train_df['cancer'] = 0

RAM_CHECK = True
DEBUG = False
patient_filter = list(sorted((set(train_df.patient_id.unique()))))
train_df = train_df[train_df.patient_id.isin(patient_filter)]

cfg = load_config('config/default.yaml')
# split = StratifiedGroupKFold(cfg.folds)
# for k, (_, test_idx) in enumerate(split.split(train_df, train_df.cancer, groups=train_df.patient_id)):
#     train_df.loc[test_idx, 'split'] = k
# train_df.split = train_df.split.astype(int)

print(f'Len df : {len(train_df)}')

train_df['fns'] = train_df['patient_id'].astype(str) + '/' + train_df['image_id'].astype(str) + '.dcm'

print(train_df.head())
y_pred = train_df['cancer'].values
print(type(y_pred))

SAVE_SIZE = int(cfg.image_size * 1.125)
SAVE_FOLDER = 'data/gen_train/'
Path(SAVE_FOLDER).mkdir(parents=True, exist_ok=True)
N_CHUNKS = len(train_df['fns']) // 2000 if len(train_df['fns']) > 2000 else 1
CHUNKS = [(len(train_df['fns']) / N_CHUNKS * k, len(train_df['fns']) /  N_CHUNKS * (k+1)) for k in range(N_CHUNKS)]
CHUNKS = np.array(CHUNKS).astype(int)
JPG_FOLDER = 'data/jpg/'

for ttt, chunk in enumerate(CHUNKS):
    print(f'chunk {ttt} of {len(CHUNKS)} chunks')
    Path(JPG_FOLDER).mkdir(parents=True, exist_ok=True)
    _ = Parallel(n_jobs=2)(delayed(convert_dicom_to_jpg)(f'{DATA_FOLDER}/{img}', save_folder=JPG_FOLDER) for img in train_df['fns'].tolist()[chunk[0]: chunk[1]]
            )
    jpgfiles = glob.glob(JPG_FOLDER + '*.jpg')
    pipe = jpg_decode_pipeline(jpgfiles, batch_size=1, num_threads=2, device_id=1)
    pipe.build()
    for i, f in enumerate(tqdm(jpgfiles)):
        patient, dicom_id = f.split('/')[-1][:-4].split('_')
        dicom = pydicom.dcmread(DATA_FOLDER + f"/{patient}/{dicom_id}.dcm")
        try:
            out = pipe.run()
            # Dali -> Torch
            img = out[0][0]
            img_torch = torch.empty(img.shape(), dtype=torch.int16, device=DEVICE)
            feed_ndarray(img, img_torch, cuda_stream=torch.cuda.current_stream(device=1))
            img = img_torch.float()
            # apply dicom preprocessing
            img = process_dicom(img, dicom)
            # resize the torch image
            img = F.interpolate(img.view(1, 1, img.size(0), img.size(1)), (SAVE_SIZE, SAVE_SIZE), mode='bilinear')[0, 0]
            img = (img * 255).clip(0, 255).to(torch.uint8).cpu().numpy()
            out_file_name = SAVE_FOLDER + f"{patient}_{dicom_id}.png"
            cv2.imwrite(out_file_name, img)
        except Exception as e:
            print(i, e)
            pipe = jpg_decode_pipeline(jpgfiles[i+1:], batch_size=1, num_threads=2, device_id=1)
            pipe.build()
            continue
    shutil.rmtree(JPG_FOLDER)
print(f'DALI Raw image load complete')

            
            