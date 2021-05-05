import random
import cv2
import numpy as np
from scipy.linalg import null_space
from Descriptor import Descriptor
from tqdm import tqdm
import os

class FourPointFeature:
    def __init__(self, random_index_list, consensus_val, test_homography_matrix):
        self.test_homography_matrix = test_homography_matrix
        self.consensus_val = consensus_val
        self.random_index_list = random_index_list


def convertToHomogeneousCoord(loc_arr):
    return np.hstack((loc_arr, np.ones((loc_arr.shape[0], 1))))


'''
Bilinear Interpolation
'''


def bilinearTransform(source_pixel_location, source_image):
    pixel_row_pos = source_pixel_location[0]
    pixel_col_pos = source_pixel_location[1]
    n_rows, n_cols = source_image.shape
    if (not ((0 <= int(np.floor(pixel_row_pos)) < source_image.shape[0]) and
             (0 <= int(np.floor(pixel_col_pos)) < source_image.shape[1]))):
        return 0
    bounding_row_val = int(np.floor(pixel_row_pos))
    bounding_col_val = int(np.floor(pixel_col_pos))

    a = pixel_row_pos - bounding_row_val
    b = pixel_col_pos - bounding_col_val

    row_factor = 1 if bounding_row_val < (n_rows - 1) else -1
    col_factor = 1 if bounding_col_val < (n_cols - 1) else -1

    newPixelIntensity = (1 - a) * (1 - b) * source_image[bounding_row_val, bounding_col_val] + \
                        (1 - a) * b * source_image[bounding_row_val, bounding_col_val + col_factor] + \
                        a * (1 - b) * source_image[bounding_row_val + row_factor, bounding_col_val] + \
                        a * b * source_image[bounding_row_val + row_factor, bounding_col_val + col_factor]
    return np.round(newPixelIntensity, 0).astype(int)


'''
Selects random 4-point correspondences and returns the A matrix used to calculate homography, the nullspace of this
matrix will give the homography
'''


def generateAMatrix(source_point_list, target_point_list, points=4):
    random_index_list = random.sample(range(0, source_point_list.shape[0] - 1), points)
    A_matrix = np.zeros([points * 2, 9], dtype=np.int32)
    for index, random_index in enumerate(random_index_list):
        A_matrix[2 * index, 0] = source_point_list[random_index, 0]
        A_matrix[2 * index, 1] = source_point_list[random_index, 1]
        A_matrix[2 * index, 2] = 1
        A_matrix[2 * index, 6] = -source_point_list[random_index, 0] * target_point_list[random_index, 0]
        A_matrix[2 * index, 7] = -source_point_list[random_index, 1] * target_point_list[random_index, 0]
        A_matrix[2 * index, 8] = -target_point_list[random_index, 0]
        A_matrix[2 * index + 1, 3] = source_point_list[random_index, 0]
        A_matrix[2 * index + 1, 4] = source_point_list[random_index, 1]
        A_matrix[2 * index + 1, 5] = 1
        A_matrix[2 * index + 1, 6] = -source_point_list[random_index, 0] * target_point_list[random_index, 1]
        A_matrix[2 * index + 1, 7] = -source_point_list[random_index, 1] * target_point_list[random_index, 1]
        A_matrix[2 * index + 1, 8] = -target_point_list[random_index, 1]
    return A_matrix, random_index_list


'''
For the generated homography by selecting four correspondence points F, this function tests how good is the 
homography by applying the homography to {source_features} - F, if more than 80% feature match their predicted
and actual location within a threshold, this function returns True, accuracy. 
'''


def testHomography(distance_threshold, random_index_list, source_features, target_features, test_homography_matrix):
    total_test_points = source_features.shape[0]

    source_homogeneous_coord = convertToHomogeneousCoord(source_features)
    predicted_target_homogeneous_coord = np.matmul(test_homography_matrix, source_homogeneous_coord.T)
    predicted_target_homogeneous_coord[2, :] = np.where(predicted_target_homogeneous_coord[2, :] == 0, 1,
                                                        predicted_target_homogeneous_coord[2, :])
    predicted_target_homogeneous_coord = predicted_target_homogeneous_coord / predicted_target_homogeneous_coord[2, :]
    predicted_target_homogeneous_coord = predicted_target_homogeneous_coord.T[:, :2]
    norm = np.linalg.norm(target_features - predicted_target_homogeneous_coord, axis=1)
    count = np.sum(norm < distance_threshold)
    return count / total_test_points * 100 >= 90, count / total_test_points



def harrisFeatureMatch(source_image, target_image, k_size=3, blockSize=16, k=0.01):
    d = Descriptor(ksize=k_size, k=k, block_size=blockSize)
    src_kp, dst_kp = d.getSourceTargetMatches(src_img=source_image, dst_img=target_image)
    return src_kp, dst_kp


'''
RANSAC->
1. get the feature correspondence between source and target
2. Generate a homography by selecting 4 random correspondence
3. Check whether the generated homography lies within a specified accuracy level
4. If yes then return the homography
5. If no then go to step 2 
'''


def getHomographyMatrix(source_image, target_image, MAX_ITERATIONS=100):
    source_features, target_features = harrisFeatureMatch(source_image, target_image)
    iterations = 0
    H_list = []
    accuracy_list = []
    while iterations <= MAX_ITERATIONS:
        test_A_matrix, random_index_list = generateAMatrix(source_features, target_features)
        test_homography_matrix = null_space(test_A_matrix)[:, 0].reshape([3, 3])
        is_valid_homography, accuracy = testHomography(10, random_index_list, source_features,
                                                                         target_features,
                                                                         test_homography_matrix)
        H_list.append(test_homography_matrix)
        accuracy_list.append(accuracy)
        iterations = iterations + 1
    args = np.array(accuracy_list).argsort()[::-1]
    max_accuracy = accuracy_list[args[0]]
    print('Accuracy ----> ' + str(100 * accuracy_list[args[0]]))
    print('Homography ----> ', H_list[args[0]])
    if 100 * max_accuracy < 80:
        print('Accuracy very low ------> ' + str(max_accuracy))
        exit()
    return H_list[args[0]]


def getImageList(path):
    imgs = []
    for root, dirs, files in os.walk(path):
        print(root)
        for file in sorted(files, key=lambda x: int(x.split(".")[0].split('-')[-1])):
            imgs.append(cv2.imread(root + '/' + file, 0))
    return imgs



def getImageHomogeneousCoord(img_shape):
    H, W = img_shape
    return np.array([[curr_row, curr_col, 1] for curr_row in range(H) for curr_col in range(W)]).astype(np.float32).T

def returnValidCoord(curr_img_arr, canvas_arr, img_shape):
    n_rows, n_cols = img_shape
    row_correct_index = np.logical_and(curr_img_arr[:, 0] > 0, curr_img_arr[:, 0] < n_rows)
    col_correct_index = np.logical_and(curr_img_arr[:, 1] > 0, curr_img_arr[:, 1] < n_cols)
    correct_index = np.logical_and(row_correct_index, col_correct_index)
    return curr_img_arr[correct_index].astype(np.int64), canvas_arr[correct_index].astype(np.int64)


def getSourceImgCoord(H, canvas_coord, img_shape):
    trg_img_coord = np.matmul(H, canvas_coord)
    trg_img_coord[2, :] = np.where(trg_img_coord[2, :] == 0, 1, trg_img_coord[2, :])
    trg_img_coord = trg_img_coord / trg_img_coord[2, :]
    curr_img_arr, canvas_arr = returnValidCoord(curr_img_arr=trg_img_coord.T[:, :2], canvas_arr=canvas_coord.T,
                                                img_shape=img_shape)
    return (np.round(curr_img_arr[:, 0]), np.round(curr_img_arr[:, 1])), (np.round(canvas_arr[:, 0]), np.round(canvas_arr[:, 1]))

def getIntensityForCanvas(curr_H, curr_img, current_pixel_location):
    curr_img_location = np.matmul(curr_H, current_pixel_location).T
    curr_img_location = curr_img_location / np.array([curr_img_location[:, 2]]).T
    intensity = bilinearTransform(curr_img_location[0], curr_img)
    return intensity


f_name = 'yard'
path = r'./input/' + f_name
imgs = getImageList(path)
H, W = imgs[0].shape
canvas_H = round(H * 1.5)
canvas_W = round(W * len(imgs))
canvas = np.zeros((canvas_H, canvas_W))

offset = round((H * len(imgs))*0.5)

H_arr = [np.eye(3)]

for i in range(1, len(imgs)):
    H_arr.append(getHomographyMatrix(imgs[i - 1], imgs[i]))

for i in range(1, len(imgs)):
    H_arr[i] = np.matmul(H_arr[i], H_arr[i - 1])

canvas_list = []
canvas_coord = getImageHomogeneousCoord(canvas.shape)

for i in tqdm(range(len(imgs))):
    canvas_curr = np.zeros(canvas.shape)
    H_curr = H_arr[i]
    img_curr = imgs[i]
    curr_img_coord, curr_img_canvas_coord = getSourceImgCoord(H=H_curr, canvas_coord=canvas_coord, img_shape=img_curr.shape)
    canvas_curr[curr_img_canvas_coord] = img_curr[curr_img_coord]
    canvas_list.append(canvas_curr)
canvas_3D = np.dstack(canvas_list)
nonzero_count = np.count_nonzero(canvas_3D, axis=2)
nonzero_count = np.where(nonzero_count == 0, 1, nonzero_count)
canvas = canvas_3D.sum(axis=2)/nonzero_count

op_f_name = 'output/canvas_' + f_name+'.png'

cv2.imwrite(op_f_name, canvas)





# cv2.imwrite('output/canvas_yard.png', canvas)


# for curr_row in tqdm(range(0, canvas_H)):
#     for curr_col in range(0, canvas_W):
#         current_pixel_location = np.array([[curr_row - offset, curr_col, 1]]).T
#         intensity_arr = np.zeros((len(imgs),))
#         for i in range(len(imgs)):
#             intensity_arr[i] = getIntensityForCanvas(H_arr[i], imgs[i], current_pixel_location)
#         canvas[curr_row, curr_col] = intensity_arr.sum()/np.count_nonzero(intensity_arr)
#
# cv2.imwrite('output/canvas_yard.png', canvas)

#get all coordinates of canvas image
#apply homography and get coordinates of the source image
#remove invalid coordinates
#use this corrdinates and populate a zero canvas image with source image intensities
#stack all these canvas arrays on top of each other to create a 3D array
#use non_zero_count to divide and np.sum over axis=2 to get canvas

