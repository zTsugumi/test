from absl import flags

FLAGS = flags.FLAGS

# siamese_ssr, siamese_ori
flags.DEFINE_string('type', 'siamese_ori', 'Choose from : {resnet, pose_only, siamese_ssr, siamese_ori}')

# General flags
flags.DEFINE_string('model', 'vanilla', 'Choose from: {vanilla, flow}.')
flags.DEFINE_string('mode', 'train', 'Choose from: {train, test}.')
flags.DEFINE_boolean('supervised', True, '')
flags.DEFINE_string('checkpoint_dir', 'capsnet\\checkpoints\\siamese_ori\\celeb_a',
                    'Path to directory for checkpoints.')
flags.DEFINE_string('logs_dir', 'capsnet\\logs\\siamese_ori\\celeb_a', 'Path to directory for tensorboard logs.')

# Dataset flags
flags.DEFINE_string('dataset', 'celeb_a', 'Choose from: {mnist, celeb_a, 300w_lp, vggface2}.')
flags.DEFINE_string('data_dir', 'dataset\\celeb_a', 'Path to directory of all datasets.')
flags.DEFINE_boolean('data_augmentation', False, 'Data augmentation.')
flags.DEFINE_boolean('data_gen', False, '')
flags.DEFINE_integer('data_dim', 64, 'Data dimension.')
flags.DEFINE_integer('batch_size', 8, '')
flags.DEFINE_float('train_size', 0.7, 'Training size')
flags.DEFINE_float('val_size', 0.15, 'Validation size')
flags.DEFINE_float('test_size', 0.15, 'Test size')

# Training flags
flags.DEFINE_string('optimizer', 'adamax', 'Choose from: {adam, sgd, adamax}.')
flags.DEFINE_float('lr', 1e-3, 'Learning rate')
flags.DEFINE_integer('lr_decay_after_step', 0, '')
flags.DEFINE_integer('lr_decay_steps', 0, '')
flags.DEFINE_string('lr_decay_type', 'exponential', 'Choose from: {none, linear, exponential}.')
flags.DEFINE_float('lr_decay_rate', 0.96, 'Decay rate for exponential learning rate decay')
flags.DEFINE_integer('n_epoch', 50, 'Total number of epochs.')
flags.DEFINE_boolean('train_eager', False, 'Eager execution')
flags.DEFINE_boolean('restart_training', False, '')
flags.DEFINE_boolean('prog_bar', True, '')
flags.DEFINE_string('loss_pose', 'mse', 'Choose from: \{mse\}.')
flags.DEFINE_float('loss_pose_weight', 1.0, 'Loss weight')
flags.DEFINE_string('loss_orth', 'orth', 'Choose from: \{orth\}.')
flags.DEFINE_float('loss_orth_weight', 0.1, 'Loss weight')

# flags.DEFINE_string('loss_fe', 'cross_entropy', '')
# flags.DEFINE_string('loss_fe', 'margin', '')
flags.DEFINE_string('loss_fe_verif', 'contrastive', '')
flags.DEFINE_float('loss_fe_verif_weight', 0.5, '')
flags.DEFINE_string('loss_fe_iden', 'margin', '')
flags.DEFINE_float('loss_fe_iden_weight', 1.0, '')

# Model flags
flags.DEFINE_float('pose_lambda_d', 1.0, 'Head pose: Control Delta')
flags.DEFINE_list('pose_stage_base', [3, 3, 3], 'Head pose: # bins in each stage')
flags.DEFINE_integer('pose_n_caps1', 7*3, 'Head pose: # primary capsule')
flags.DEFINE_integer('pose_n_caps1_dim', 5, 'Head pose: primary capsule dimensionality')
flags.DEFINE_integer('pose_n_caps2', 3, 'Head pose: # class capsule')
flags.DEFINE_integer('pose_n_caps2_dim', 16, 'Head pose: class capsule dimensionality')
flags.DEFINE_integer('pose_routing', 2, 'Head pose: # capsule routing')

flags.DEFINE_integer('fe_n_class', 10177, 'Number of identity')
flags.DEFINE_integer('fe_n_caps1', 32, 'Head pose: # primary capsule')
flags.DEFINE_integer('fe_n_caps1_dim', 8, 'Head pose: primary capsule dimensionality')
flags.DEFINE_integer('fe_n_caps2', 10, 'Head pose: # class capsule')
flags.DEFINE_integer('fe_n_caps2_dim', 16, 'Head pose: class capsule dimensionality')
flags.DEFINE_integer('fe_routing', 3, 'Head pose: # capsule routing')
