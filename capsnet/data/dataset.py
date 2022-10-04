from capsnet.data import the300w_lp, celeb_a, mnist


def make_dataset_iterator(
    data_name,
    data_dir,
    batch_size,
    data_dim,
    mode='train',
    supervised=True,
    data_gen=False,
    data_augmentation=False,    # WIP, for now default False
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
):
  assert train_size + val_size + test_size == 1, 'Data split error'

  if data_name == '300w_lp':
    dataset = the300w_lp.make_dataset(
        data_dir, mode, batch_size, (train_size, val_size, test_size))
  elif data_name == 'celeb_a':
    dataset = celeb_a.make_dataset(
        data_dir, data_dim, mode, batch_size, (train_size, val_size, test_size))
  elif data_name == 'mnist':
    dataset = mnist.make_dataset(
        data_dir, data_dim, mode, batch_size)
  else:
    raise ValueError(f'Data {data_name} not recognized')

  return dataset
