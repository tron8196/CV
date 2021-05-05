import os
import numpy as np

def getImageHomogeneousCoord(img_shape):
    H, W = img_shape
    return np.array([[curr_row, curr_col, 1] for curr_row in range(H) for curr_col in range(W)]).astype(np.float32).T

def returnValidCoord(curr_arr, max_len):
    return curr_arr[np.logical_and(curr_arr >= 0, curr_arr < max_len)]

def getSourceImgCoord(H, canvas_coord, img_shape):
    n_rows, n_cols = img_shape
    trg_img_coord = np.matmul(H, canvas_coord)
    trg_img_coord[2, :] = np.where(trg_img_coord[2, :] == 0, 1, trg_img_coord[2, :])
    trg_img_coord = trg_img_coord / trg_img_coord[2, :]
    row_arr = trg_img_coord[0, :]
    col_arr = trg_img_coord[1, :]
    row_arr = returnValidCoord(row_arr, n_rows)
    col_arr = returnValidCoord(col_arr, n_cols)
    return np.round(row_arr).astype(np.int64), np.round(col_arr).astype(np.int64)






H = np.random.rand(3, 3)
# print(H.shape)

a = np.random.randint(0, 255, (3, 3))

d = np.random.randint(0, 255, (100, 100))
coord = getImageHomogeneousCoord(a.shape)
# print(coord.shape)
args = getSourceImgCoord(H, coord, a.shape)
print(a)
print(args)
print(a[args])







