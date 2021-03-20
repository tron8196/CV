# d is num of color channels
import numpy as np
import cv2


def gaussianPDF(X, mu, sigma):
    scale_factor = 1 / (sigma * np.sqrt(2 * np.pi))
    gaussian = scale_factor * np.exp(-(1 / 2) * ((X - mu) / sigma) ** 2)
    return gaussian


class backgroundSubtractionNonParametric:
    pixel_covariance_matrix = None
    video_frame_array = None

    def __init__(self, window_size=5, frame_read_per_second=10):
        self.window_size = window_size
        self.frame_read_per_second = frame_read_per_second
        self.color_channels = self.n_rows = self.n_cols = self.num_frames = None

    def videoInfo(self):
        pass

    def readVideo(self, video_file_name):
        video_capture = cv2.VideoCapture(video_file_name)
        frame_list = []
        count = 0
        last_frame = None
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if count % self.frame_read_per_second == 0 and frame is not None:
                # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                print(frame.shape)
                frame_list.append(frame)
            if frame is None:
                break
            count = count + 1

        frame_tuple = tuple(frame_list)
        frame_3D_array = np.dstack(frame_tuple)
        self.video_frame_array = frame_3D_array
        print(frame_3D_array.shape)
        self.n_rows, self.n_cols, self.color_channels, self.num_frames = frame_3D_array.shape

    def backgroundSubtract(self):
        pass

a = backgroundSubtractionNonParametric()
a.readVideo('Jump.avi')