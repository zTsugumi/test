# WIP
import os
from capsnet.utils import ops_file


def make_dataset(dir, untar=False):
  data_path = os.path.join(dir, 'data')
  meta_path = os.path.join(dir, 'meta')

  if untar:
    for filename in os.listdir(data_path):
      f = os.path.join(data_path, filename)
      # checking if it is a tar file
      if os.path.isfile(f) and f.endswith('.tar.gz'):
        tarname = filename.split('.')[0]
        ops_file, untar(tarname, f, data_path)

