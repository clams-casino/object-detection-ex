import numpy as np

import cv2 # TODO remove later

from agent import PurePursuitPolicy
from utils import launch_env, seed, makedirs, display_seg_mask, display_img_seg_mask

npz_index = 0
def save_npz(img, boxes, classes):
    global npz_index
    with makedirs("./data_collection/dataset"):
        np.savez(f"./data_collection/dataset/{npz_index}.npz", *(img, boxes, classes))
        npz_index += 1

def clean_segmented_image(seg_img):
    # TODO
    # Tip: use either of the two display functions found in util.py to ensure that your cleaning produces clean masks
    # (ie masks akin to the ones from PennFudanPed) before extracting the bounding boxes

    # TODO do not put bounding box and label if the bounding box is too small

    pass
    # return boxes, classes

seed(123)
environment = launch_env()

policy = PurePursuitPolicy(environment)

MAX_STEPS = 500

while True:
    obs = environment.reset()
    environment.render(segment=True)
    rewards = []

    nb_of_steps = 0

    while True:
        action = policy.predict(np.array(obs))

        obs, rew, done, misc = environment.step(action) # Gives non-segmented obs as numpy array
        segmented_obs = environment.render_obs(True)  # Gives segmented obs as numpy array

        rewards.append(rew)
        # environment.render(segment=int(nb_of_steps / 50) % 2 == 0)


        # obs_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        # print('colored image shape')
        # print(obs.shape) # TODO note that the image from the simulator is not square
        # print('')
        # cv2.imshow('observation',obs_bgr)
        # cv2.waitKey(1)

        segmented_obs_bgr = cv2.cvtColor(segmented_obs, cv2.COLOR_RGB2BGR)
        print('segmentation image shape')
        print(segmented_obs.shape) # TODO note that the image from the simulator is not square
        print('')
        cv2.imshow('segmentation',segmented_obs_bgr)
        cv2.waitKey(1)

        ''' THE SEGMENTED IMAGES IS ACTUALLY RGB,
         where the segmented classes have the same color always
        
        0 - background -> rgb [255, 0, 255] -> bgr [255, 0, 255]

        1 - duckie -> rgb [100, 117, 226] -> bgr [226, 117, 100]

        2 - cone -> rgb [226, 111, 101] -> bgr [101, 111, 226]

        3 - truck -> rgb [116, 114, 117] -> bgr [117, 114, 116]

        4 - bus -> rgb [216, 171, 15] -> bgr [15, 171, 216]

        '''

        # TODO boxes, classes = clean_segmented_image(segmented_obs)


        # TODO only save an image that's only background with some probability


        # TODO save_npz(obs, boxes, classes)

        nb_of_steps += 1

        if done or nb_of_steps > MAX_STEPS:
            break
