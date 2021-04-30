import numpy as np
import cv2


def gaussianPDF(X, mu, sigma):
    scale_factor = 1 / (sigma * np.sqrt(2 * np.pi))
    gaussian = scale_factor * np.exp(-(1 / 2) * ((X - mu) / sigma) ** 2)
    return gaussian


def roundArray(arr, precision=5):
    return np.round(arr, precision)


class backgroundSubtractionGMM:
    learning_rate = 0.3
    number_of_gaussians = 5
    background_threshold = 0.6
    weights_array = None
    mean_array = None
    variance_array = None
    n_rows = n_cols = num_frames = None
    video_frame_array = None
    frame_read_per_second = 10
    MAX_ITERATIONS = None
    per_frame_background_array = None

    def __init__(self, frame_per_second, number_of_gaussians=3):
        self.number_of_gaussians = number_of_gaussians
        self.frame_read_per_second = frame_per_second

    # Read video from the specified path and load it in the video_frame_array attribute
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
        self.video_frame_array = frame_3D_array
        self.n_rows, self.n_cols, self.num_frames = frame_3D_array.shape
        self.setupWeightsAndMeanArray()

    # Initialize array weights for mean and standard deviation
    def setupWeightsAndMeanArray(self):
        self.weights_array = np.ones([self.n_rows, self.n_cols, self.number_of_gaussians])
        sum_weights_array = np.sum(self.weights_array, axis=2)
        self.weights_array = roundArray(self.weights_array / sum_weights_array[:, :, np.newaxis])
        self.mean_array = np.zeros(self.weights_array.shape)
        self.per_frame_background_array = np.ones(self.video_frame_array.shape) * 255

        for row_index in range(self.n_rows):
            for col_index in range(self.n_cols):
                pixel_intensity_array = self.video_frame_array[row_index, col_index, :]
                self.mean_array[row_index, col_index, :] = np.random.choice(pixel_intensity_array,
                                                                            self.number_of_gaussians)

        self.variance_array = 0.02 * self.mean_array + 1e-8

    def resetNonMatchingGaussian(self, row_index, col_index, pixel_intensity, likelihood, matched_gaussian):
        if np.sum(matched_gaussian) == 0:
            argmin_gaussian = np.argmin(likelihood)
            self.mean_array[row_index, col_index, argmin_gaussian] = pixel_intensity
            self.variance_array[row_index, col_index, argmin_gaussian] = 0.02 * pixel_intensity

    def getMatchedGaussianIndex(self, row_index, col_index, pixel_intensity, threshold_variance=2.5):
        gaussian_mean = self.mean_array[row_index, col_index, :]
        gaussian_variance = self.variance_array[row_index, col_index, :]
        upper_bound = gaussian_mean + threshold_variance * gaussian_variance
        lower_bound = gaussian_mean - threshold_variance * gaussian_variance
        return np.logical_and(pixel_intensity > lower_bound, pixel_intensity < upper_bound)

    def isBackgroundPixel(self, omegaBySigma):
        omegaBySigma[::-1].sort()
        return np.sum(omegaBySigma[:self.number_of_gaussians - 2]) >= self.background_threshold

    # The following is the exact implementation of the Stauffer and Grimson Paper on adaptive background subtraction
    def detectBackground(self, MAX_ITERATIONS=200):
        self.MAX_ITERATIONS = MAX_ITERATIONS
        for time_index in range(self.num_frames):
            print('Completed For Frame number ' + str(time_index + 1))
            for row_index in range(self.n_rows):
                for col_index in range(self.n_cols):
                    pixel_intensity_array = self.video_frame_array[row_index, col_index, :].astype(np.float)
                    curr_mean_array = self.mean_array[row_index, col_index, :]
                    curr_variance_array = self.variance_array[row_index, col_index, :]
                    curr_weights_array = self.weights_array[row_index, col_index, :]

                    pixel_intensity = self.video_frame_array[row_index, col_index, time_index]
                    likelihood = np.array([gaussianPDF(pixel_intensity, curr_mean_array[gaussian_index],
                                                       curr_variance_array[gaussian_index])
                                           for gaussian_index in range(self.number_of_gaussians)])
                    matched_gaussians = self.getMatchedGaussianIndex(row_index, col_index, pixel_intensity)
                    self.resetNonMatchingGaussian(row_index, col_index, pixel_intensity, likelihood, matched_gaussians)
                    curr_weights_array = (
                                                     1 - self.learning_rate) * curr_weights_array + self.learning_rate * matched_gaussians
                    curr_weights_array = roundArray(curr_weights_array / np.sum(curr_weights_array))
                    self.weights_array[row_index, col_index, :] = curr_weights_array
                    learning_rate_rho = self.learning_rate * likelihood
                    updated_mean_array = np.copy(curr_mean_array)
                    updated_variance_array = np.copy(curr_variance_array)
                    updated_mean_array[matched_gaussians] = \
                    ((1 - learning_rate_rho) * curr_mean_array + learning_rate_rho * pixel_intensity)[matched_gaussians]
                    updated_variance_array[matched_gaussians] = (
                                (1 - learning_rate_rho) * curr_variance_array + learning_rate_rho * (
                                    pixel_intensity - curr_mean_array) ** 2)[matched_gaussians]
                    self.mean_array[row_index, col_index, :] = updated_mean_array
                    self.variance_array[row_index, col_index, :] = updated_variance_array

                    if self.isBackgroundPixel(roundArray(curr_weights_array / curr_variance_array)):
                        self.per_frame_background_array[row_index, col_index, time_index] = 0

'''
parameters
number_of_gaussians, DEFAULT = 3, this specifies the number of default gaussians in the mixture model
frame_per_second, DEFAULT = 10
'''
a = backgroundSubtractionGMM(5, 10)
a.readVideo('Run.avi')
a.detectBackground()

#Will write the detected foreground images in the output/ folder with gmm_background_{frame_id}.png name
for i in range(a.num_frames):
    cv2.imwrite('./output/gmm_background_' + str(i) + '.png', a.per_frame_background_array[:, :, i])
