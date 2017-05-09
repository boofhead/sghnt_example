import numpy as np
import os
import urllib.request
import gzip
import struct


def download_data(url, force_download=True):
    if not os.path.isdir('data'):
        os.makedirs('data')
    fname = os.path.join('data', url.split("/")[-1])
    if force_download or not os.path.exists(fname):
        urllib.request.urlretrieve(url, fname)
    return fname


def read_data(label_url, image_url):
    with gzip.open(download_data(label_url)) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(download_data(image_url), 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label, image)

