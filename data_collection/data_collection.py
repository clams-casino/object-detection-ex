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


def filter_boxes(boxes, min_area=None, min_aspect_ratio=None):
    valid_boxes = []
    for box in boxes:
        w = box[2] - box[0]
        h = box[3] - box[1]

        if min_area is not None:
            if w*h < min_area:
                continue
        if min_aspect_ratio is not None:
            if h/w < min_aspect_ratio:
                continue
        valid_boxes.append(box)

    return valid_boxes


def get_bounding_boxes(binary_seg_img):
    '''
        Takes a binary segmented image of a single class and returns a list of the bounding boxes
    '''
    contours, _ = cv2.findContours(binary_seg_img, cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            boxes.append( np.array([x, y, x+w, y+h]) )
            
    return boxes


def draw_bounding_boxes(img, boxes):
    for box in boxes:
        img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255,255,255), thickness=1)
    return img


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
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(1.0*kernel_size),int(1.0*kernel_size))))

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

NN_IMG_SIZE = (224,224)

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


        background_seg = _get_background(segmented_obs)
        obs = cv2.resize(obs, NN_IMG_SIZE)
        background_seg = cv2.resize(background_seg, NN_IMG_SIZE)
        background_bb = get_bounding_boxes(background_seg)
        obs_background_bb = draw_bounding_boxes(obs.copy(), background_bb)

        duckie_seg = _get_duckies(segmented_obs)
        obs = cv2.resize(obs, NN_IMG_SIZE)
        duckie_seg = cv2.resize(duckie_seg, NN_IMG_SIZE)
        duckie_bb = filter_boxes(get_bounding_boxes(duckie_seg))
        obs_duckie_bb = draw_bounding_boxes(obs.copy(), duckie_bb)

        cone_seg = _get_cones(segmented_obs)
        obs = cv2.resize(obs, NN_IMG_SIZE)
        cone_seg = cv2.resize(cone_seg, NN_IMG_SIZE)
        cone_bb = filter_boxes(get_bounding_boxes(cone_seg))
        obs_cone_bb = draw_bounding_boxes(obs.copy(), cone_bb)

        truck_seg = _get_trucks(segmented_obs)
        obs = cv2.resize(obs, NN_IMG_SIZE)
        truck_seg = cv2.resize(truck_seg, NN_IMG_SIZE)
        truck_bb = get_bounding_boxes(truck_seg)
        obs_truck_bb = draw_bounding_boxes(obs.copy(), truck_bb)

        bus_seg = _get_buses(segmented_obs)
        obs = cv2.resize(obs, NN_IMG_SIZE)
        bus_seg = cv2.resize(bus_seg, NN_IMG_SIZE)
        bus_bb = get_bounding_boxes(bus_seg)
        obs_bus_bb = draw_bounding_boxes(obs.copy(), bus_bb)

        obs_bounding_boxes = obs_background_bb
        class_seg = background_seg

        side_by_side = np.hstack((obs_bounding_boxes, cv2.applyColorMap(class_seg, cv2.COLORMAP_SPRING)))
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
