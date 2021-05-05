import numpy as np
import cv2
from scipy.spatial.distance import cdist
from skimage.feature import corner_peaks

#Add z-coord as 1 for all points to convert to homogeneous coordinate system
def convertToHomogeneousCoord(loc_arr):
    return np.hstack((loc_arr, np.ones((loc_arr.shape[0], 1))))

class Descriptor:
    def __init__(self, ksize=3, k=0.04, block_size=16):
        self.ksize = ksize
        self.k = k
        self.block_size = block_size

    #Get Harris Corners for an image
    def getKeypoints(self, img):
        # H, W = img.shape
        # M = cv2.cornerEigenValsAndVecs(src=img, ksize=self.ksize, blockSize=self.ksize)
        # response = np.zeros(img.shape, dtype=np.float32)
        # for i in range(H):
        #     for j in range(W):
        #         lambda1 = M[i, j, 0]
        #         lambda2 = M[i, j, 1]
        #         response[i, j] = lambda1 * lambda2 - self.k * ((lambda2 + lambda1) ** 2)
        #
        Ix = cv2.Sobel(src=img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=self.ksize)
        Iy = cv2.Sobel(src=img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=self.ksize)
        Ixx = Ix * Iy
        Iyy = Iy * Iy
        Ixy = Ix * Iy
        response = Ixx * Iyy - self.k * (Ixy ** 2)
        args = corner_peaks(response, threshold_rel=self.k, exclude_border=8)
        return args


    '''
    Define a descriptor around a harris corner detected for a corner, by normalizing the array by setting it to mean 0 and 
    variance 1.
    '''
    def getKeyPointsDescriptor(self, img, keypoints):
        patch_size = self.block_size
        img.astype(np.float32)
        desc = []
        for i, kp in enumerate(keypoints):
            y, x = kp
            patch = img[y - (patch_size // 2):y + ((patch_size + 1) // 2),
                    x - (patch_size // 2):x + ((patch_size + 1) // 2)]
            desc.append(self.feature_descriptor(patch))
        return np.array(desc)

    def feature_descriptor(self, patch):
        std_dev = np.round(patch.std(), 3)
        mean = np.round(patch.mean(), 3)
        patch = (patch - mean) / (std_dev if std_dev != 0 else 1)
        feature = patch.reshape(patch.shape[0] * patch.shape[1])
        return feature


    ''' 
    Find distance between all pairs of source and target points and find which target point lies closest to a source point
    additionally we enforce another constraint that the second closest point should be further than a threshold.
    '''
    def match_descriptors(self, desc1, desc2, threshold=0.5):
        N = desc1.shape[0]
        dists = cdist(desc1, desc2)
        arg = dists.argsort(axis=1)
        dists_sorted = np.sort(dists, axis=1)
        src_match = np.where(dists_sorted[:, 0] / dists_sorted[:, 1] < threshold)[0]
        dst_match = (arg[src_match])[:, 0]
        matches = np.vstack((src_match, dst_match)).T
        return matches

    '''
    Given two images find source and target matching key-points
    '''
    def getSourceTargetMatches(self, src_img, dst_img):
            src_keypoint = self.getKeypoints(src_img)
            dst_keypoint = self.getKeypoints(dst_img)
            src_descriptors = self.getKeyPointsDescriptor(src_img, src_keypoint)
            dst_descriptors = self.getKeyPointsDescriptor(dst_img, dst_keypoint)
            matches = self.match_descriptors(src_descriptors, dst_descriptors)
            return src_keypoint[matches[:, 0]], dst_keypoint[matches[:, 1]]