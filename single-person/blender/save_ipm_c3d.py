import numpy as np
import pickle
import ezc3d
import os

def save_ipm_c3d(name, main_paths, ipm_sts, rod_len, cart_h):
#name: '1'
#main_path = ['c3d_ipm/exp_1/, points_ipm/exp_1/]
    for path in main_paths:
        if os.path.exists(path) == False:
            os.mkdir(path)
    num_frames =  ipm_sts.shape[0]
    end_rod = np.zeros((num_frames, 3))
    for k in range(num_frames):
        x, y, theta, phi = ipm_sts[k, 0], ipm_sts[k, 1], ipm_sts[k, 2], ipm_sts[k, 3]
        end_of_pendulum_local = rod_len * np.array(
            [[np.sin(theta)], [-np.cos(theta) * np.sin(phi)], [np.cos(theta) * np.cos(phi)]])
        end_of_pendulum = np.array([[x], [y], [cart_h]]) + end_of_pendulum_local
        end_rod[k, :] = end_of_pendulum.reshape(3)
    ipm_c3d_cart = np.concatenate((ipm_sts[:,:2], np.ones((num_frames, 1)) * cart_h), axis=-1)
    ipm_c3d = np.stack((ipm_c3d_cart, end_rod), axis=1)
    ipm_c3d_swap = ipm_c3d[:, :, [0, 2, 1]]
    points_path = main_paths[1] + name + '.npy'
    with open(points_path, 'wb') as f:
        np.save(f, ipm_c3d_swap)

    c3d = ezc3d.c3d()

    c3d['header']['points']['size'] = 2
    c3d['header']['points']['frame_rate'] = 24
    c3d['header']['points']['first_frame'] = 1
    c3d['header']['points']['last_frame'] = num_frames
    c3d['header']['analogs']['size'] = 0
    c3d['header']['analogs']['frame_rate'] = 0.0
    c3d['header']['analogs']['first_frame'] = 1
    c3d['header']['analogs']['last_frame'] = - 1

    c3d['parameters']['POINT']['USED']['value'] = np.array([2])
    c3d['parameters']['POINT']['RATE']['value'] = np.array([24])
    c3d['parameters']['POINT']['FRAMES']['value'] = np.array([num_frames])
    c3d['parameters']['POINT']['LABELS']['value'] = ['cart', 'rod']
    c3d['parameters']['POINT']['UNITS']['value'] = ['m']


    new_points = np.ones((4, 2, num_frames))
    new_points[:-1] = np.transpose(ipm_c3d, (2, 1, 0))
    c3d['data']['points'] = new_points
    c3d['data']['meta_points']['residuals'] = np.zeros((1, 2, num_frames))
    c3d['data']['meta_points']['camera_masks'] = np.ones((7, 2, num_frames)) > 2
    c3d_path = main_paths[0] + name + '.c3d'
    c3d.write(c3d_path)
