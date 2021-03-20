import cv2
import numpy as np


def roundFloatVal(val, precision=3):
    return np.round(val, precision)


class ImageWindowRegion:

    def __init__(self, window_id, image_window_arr, num_random_pairs=10, isBackgroundFlag=None):
        self.window_id = window_id
        self.image_window_arr = image_window_arr
        self.isBackgroundFlag = isBackgroundFlag

    def isBackgroundRegion(self):
        return self.isBackgroundFlag

    def getImageWindowArr(self):
        return self.image_window_arr

    def setBackgroundFlag(self, isBackground):
        self.isBackgroundFlag = isBackground


class OrderBasedBackgroundSubtraction:
    video_frame_array = n_rows = n_cols = num_frames = None
    image_window_list = []
    image_window_regions_dict = {}
    detection_threshold = 0.03

    def __init__(self, window_size, frame_read_per_second=10):
        self.window_size = window_size
        self.frame_read_per_second = frame_read_per_second

    def readVideo(self, video_file_name):
        video_capture = cv2.VideoCapture(video_file_name)
        frame_list = []
        count = 0
        last_frame = None
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if count % self.frame_read_per_second == 0 and frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                frame_list.append(gray)
            if frame is None:
                break
            count = count + 1

        frame_tuple = tuple(frame_list)
        frame_3D_array = np.dstack(frame_tuple)
        print(frame_3D_array.shape)
        self.video_frame_array = frame_3D_array
        self.n_rows, self.n_cols, self.num_frames = frame_3D_array.shape

    def setupImageWindowRegionList(self):
        window_id = 1
        for start_row_index in range(0, self.n_rows, self.window_size):
            for start_col_index in range(0, self.n_cols, self.window_size):
                end_row_index = start_row_index + self.window_size
                end_col_index = start_col_index + self.window_size
                self.image_window_list.append(np.array([(i, j) for i in range(start_col_index, end_col_index) for j in
                                                        range(start_row_index, end_row_index) if
                                                        i < self.n_rows and j < self.n_cols]))
                curr_image_window_arr = np.array([(i, j) for i in range(start_col_index, end_col_index) for j in
                                                  range(start_row_index, end_row_index) if
                                                  i < self.n_rows and j < self.n_cols])
                if curr_image_window_arr.size == 0:
                    break
                # print(curr_image_window_arr.shape)
                curr_image_window_region = ImageWindowRegion(window_id, curr_image_window_arr)
                self.image_window_regions_dict[window_id] = curr_image_window_region
                window_id = window_id + 1
        self.getRandomLocationFromImagePatch(10)

    def getRandomLocationFromImagePatch(self, window_id):
        currImageWindowRegion = self.image_window_regions_dict[window_id]
        curr_image_window_arr = currImageWindowRegion.getImageWindowArr()

        num_locations = curr_image_window_arr.shape[0]

        numerator_num_pair_choice = np.random.randint(1, num_locations * 0.1)
        denominator_pair_choice = np.random.randint(numerator_num_pair_choice, num_locations * 0.2)

        numerator_location_list = curr_image_window_arr[np.random.choice(num_locations,
                                                                         numerator_num_pair_choice, replace=False), :]

        denominator_location_list = curr_image_window_arr[np.random.choice(num_locations,
                                                                           denominator_pair_choice, replace=False), :]

        return numerator_location_list, denominator_location_list

    def writeCurrentBackgroundSubtractionImage(self, curr_frame_index):
        curr_image = np.zeros((self.n_rows, self.n_cols))

        for window_id in self.image_window_regions_dict.keys():
            if not self.image_window_regions_dict[window_id].isBackgroundRegion():
                location_list = self.image_window_regions_dict[window_id].getImageWindowArr()
                curr_image[tuple(np.transpose(location_list))] = 255
        cv2.imwrite('./output/background' + str(curr_frame_index) + '.png', curr_image)

    def getBackground(self):
        self.setupImageWindowRegionList()
        for curr_frame_index in range(0, self.num_frames):
            for window_id in self.image_window_regions_dict.keys():
                last_frame_index = curr_frame_index + 1
                if last_frame_index >= self.num_frames:
                    break
                curr_frame = self.video_frame_array[:, :, curr_frame_index]
                last_frame = self.video_frame_array[:, :, last_frame_index]

                numerator_location_list, denominator_location_list = self.getRandomLocationFromImagePatch(window_id)

                numerator_sum_curr_frame = curr_frame[tuple(np.transpose(numerator_location_list))].sum()
                denominator_sum_curr_frame = curr_frame[tuple(np.transpose(denominator_location_list))].sum()

                numerator_sum_last_frame = last_frame[tuple(np.transpose(numerator_location_list))].sum()

                denominator_sum_last_frame = last_frame[tuple(np.transpose(denominator_location_list))].sum()

                curr_frame_ratio = roundFloatVal(numerator_sum_curr_frame / denominator_sum_curr_frame)
                last_frame_ratio = roundFloatVal(numerator_sum_last_frame / denominator_sum_last_frame)

                if np.abs(curr_frame_ratio - last_frame_ratio) > self.detection_threshold:
                    self.image_window_regions_dict[window_id].setBackgroundFlag(False)
                else:
                    self.image_window_regions_dict[window_id].setBackgroundFlag(True)
            self.writeCurrentBackgroundSubtractionImage(curr_frame_index)


a = OrderBasedBackgroundSubtraction(10, 10)
a.readVideo('Jump.avi')
# a.setupImageWindowRegionList()
a.getBackground()
