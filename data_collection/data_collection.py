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

    # TODO have different bounding box size limits? For example since cones are smaller?

    # TODO think about metrics for what's a good image to save that can capture a lot of the cases in evaluation?

    pass
    # return boxes, classes


def get_bounding_boxes(binary_seg_img):
    '''
        Takes a binary segmented image of a single class and returns a list of the bounding boxes
    '''


def _remove_snow(img, kernel_size=6, process_entire_image=True):
    '''
        Assumes img is a single channel image
        Remove snow using erode, then restort using dilate
    '''
    if process_entire_image:
        start_height = 0
    else:
        # If option is false, then do not apply to the top ~25% of the image where the snow is not present
        h = img.shape[0]
        start_height = int(np.floor(0.3*h))

    # open to remove snow
    processed_section = cv2.morphologyEx(img[start_height:,:], cv2.MORPH_OPEN, 
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size)))

    # close to stitch together seperated contours at distances
    processed_section = cv2.morphologyEx(processed_section, cv2.MORPH_CLOSE, 
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(0.5*kernel_size),int(0.5*kernel_size))))

    img[start_height:,:] = processed_section  #for duckies, some speckles remain, but can be filtered out by bounding box size

    return img


def _get_background(seg_img):
    return cv2.inRange(seg_img, (255, 0, 255), (255, 0, 255))

def _get_duckies(seg_img):
    return _remove_snow(cv2.inRange(seg_img, (100, 116, 226), (100, 118, 226)), kernel_size=6, process_entire_image=False) #kepp a bit of snow for now, to get more of the duckies

def _get_cones(seg_img):
    return cv2.inRange(segmented_obs, (226, 111, 101), (226, 111, 101))

def _get_trucks(seg_img):
    return _remove_snow(cv2.inRange(segmented_obs, (116, 114, 117), (116, 114, 117)))

def _get_buses(seg_img):
    return _remove_snow(cv2.inRange(segmented_obs, (216, 171, 15), (216, 171, 15)))

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

        # segmented_obs_bgr = cv2.cvtColor(segmented_obs, cv2.COLOR_RGB2BGR)
        # print('segmentation image shape')
        # print(segmented_obs.shape) # TODO note that the image from the simulator is not square
        # print('')
        # cv2.imshow('segmentation',segmented_obs_bgr)
        # cv2.waitKey(1)

        ''' THE SEGMENTED IMAGES IS ACTUALLY RGB,
         where the segmented classes have the same color always
        
        0 - background -> rgb [255, 0, 255] 

        1 - duckie -> rgb [100, 117, 226]

        2 - cone -> rgb [226, 111, 101]

        3 - truck -> rgb [116, 114, 117]

        4 - bus -> rgb [216, 171, 15]

        '''


        class_seg = _get_background(segmented_obs)
        # class_seg = _get_duckies(segmented_obs)
        # class_seg = _get_cones(segmented_obs)
        # class_seg = _get_trucks(segmented_obs)
        # class_seg = _get_buses(segmented_obs)
        side_by_side = np.hstack((obs, cv2.applyColorMap(class_seg, cv2.COLORMAP_SPRING)))
        cv2.imshow('segmented_class',side_by_side)
        # cv2.imshow('duckie', duckie_seg)
        cv2.waitKey(1)



        # TODO boxes, classes = clean_segmented_image(segmented_obs)

        # TODO maybe save every other images (or every nth) to avoid too many basically repeat images?

        # TODO only save an image that's only background with some probability
        # TODO save_npz(obs, boxes, classes)

        nb_of_steps += 1

        if done or nb_of_steps > MAX_STEPS:
            break
