import numpy as np


def EulerAngles2Vectors(rx, ry, rz):
  '''
  rx: roll
  ry: pitch
  rz: yaw
  '''
  ry *= -1
  R_x = np.array([[1.0, 0.0, 0.0],
                  [0.0, np.cos(rx), -np.sin(rx)],
                  [0.0, np.sin(rx), np.cos(rx)]])

  R_y = np.array([[np.cos(ry), 0.0, np.sin(ry)],
                  [0.0, 1.0, 0.0],
                  [-np.sin(ry), 0.0, np.cos(ry)]])

  R_z = np.array([[np.cos(rz), -np.sin(rz), 0.0],
                  [np.sin(rz), np.cos(rz), 0.0],
                  [0.0, 0.0, 1.0]])

  R = R_x @ R_y @ R_z

  l_vec = R @ np.array([1, 0, 0]).T
  b_vec = R @ np.array([0, 1, 0]).T
  f_vec = R @ np.array([0, 0, 1]).T
  return l_vec, b_vec, f_vec
