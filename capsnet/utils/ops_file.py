import tarfile


def untar(data_name, data_path, extract_path):
  print(f'Extracting {data_name} ...')
  tarobj = tarfile.open(data_path, 'r:gz', encoding='utf-8')
  tarobj.extractall(extract_path)
