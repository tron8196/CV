# d is num of color channels
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join

import re

'''
Takes in a n*3 2D array, where col=0 indicates current val, col=1 indicates mean and col=2 indicates standard deviation
'''


def gaussian_pdf(arr):
    return (1 / np.sqrt(2 * np.pi * arr[:, 2] ** 2)) * np.exp(-(np.square((arr[:, 0] - arr[:, 1]) / arr[:, 2])) / 2)


def roundFloatVal(val, precision=3):
    return np.round(val, precision)


def atoi(text) -> object:
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

class backgroundSubtractionNonParametric:
    pixel_covariance_matrix = None
    video_frame_array = None
    threshold = None
    frame_block_size = None

    def __init__(self, threshold=0.6, frame_block_size=10):
        self.n_rows = self.n_cols = self.num_frames = None
        self.threshold = threshold
        self.frame_block_size = frame_block_size

    def readVideo(self, video_file_name):
        video_capture = cv2.VideoCapture(video_file_name)
        frame_list = []
        count = 0
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                frame_list.append(gray)
            if frame is None:
                break
            count = count + 1

        frame_tuple = tuple(frame_list)
        frame_3D_array = np.dstack(frame_tuple)
        self.video_frame_array = frame_3D_array
        self.n_rows, self.n_cols, self.num_frames = frame_3D_array.shape

    def readVideoFromFrames(self, folder_path):
        only_files_path = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
        only_files_path.sort(key=natural_keys)
        frame_list = []
        count = 0
        for file_name in only_files_path:
            img = cv2.imread(folder_path + file_name)
            if img is not None and count % 20 == 0:
                frame_list.append(img)
            count = count + 1
        frame_tuple = tuple(frame_list)
        frame_3D_array = np.dstack(frame_tuple)
        self.video_frame_array = frame_3D_array
        self.n_rows, self.n_cols, self.num_frames = frame_3D_array.shape


    def detectBackground(self):
        eps = 1e-8
        print(self.num_frames)
        for current_frame_index in range(self.frame_block_size, self.num_frames, self.frame_block_size):
            print(current_frame_index)
            curr_image = np.zeros((self.n_rows, self.n_cols))
            for row_index in range(self.n_rows):
                for col_index in range(self.n_cols):
                    current_pixel_val = [self.video_frame_array[row_index, col_index, current_frame_index].astype(
                        np.float)] * (self.frame_block_size - 1)

                    # current_pixel_val = self.video_frame_array[row_index, col_index,
                    #                      current_frame_index - self.frame_block_size: current_frame_index].astype(
                    #     np.float)[:self.frame_block_size -1]

                    current_pixel_val = np.array(current_pixel_val)
                    previous_pixel_val = self.video_frame_array[row_index, col_index,
                                         current_frame_index - self.frame_block_size: current_frame_index].astype(
                        np.float)
                    previous_pixel_val_1 = np.zeros(previous_pixel_val.shape[0]).astype(np.float)
                    previous_pixel_val_1[1:] = previous_pixel_val[:-1]
                    std_dev = (np.abs(previous_pixel_val - previous_pixel_val_1)[1:]) / (0.68 * np.sqrt(2)) + eps
                    mean_arr = previous_pixel_val[1:]
                    pdf_stack = np.dstack((mean_arr, current_pixel_val, std_dev))[0]
                    gaussian_pdf_estimate = (1 / (self.frame_block_size - 1)) * np.sum(gaussian_pdf(pdf_stack))
                    if roundFloatVal(gaussian_pdf_estimate) < self.threshold:
                        curr_image[row_index, col_index] = 255
            cv2.imwrite('./output/non_parametric_background_' + str(current_frame_index) + '.png', curr_image)


'''
parameters
frame_block_size, DEFAULT = 5
threshold, DEFAULT = 0.7
'''
a = backgroundSubtractionNonParametric()
# a.readVideo('Jump.avi')

a.readVideoFromFrames('./input/')
a.detectBackground()
