
"""
计算下采样方法

"""
import numpy as np
import math
import numpy as np
import cv2

path = "./a.bmp"
def downsampling(width, mat):
    # width 是64
    mat_new = []
    for i in range(width/2):
        mat_new.append([])
        for j in range(width/2):
            mat_new[i].append(max(mat[i*2][j*2], mat[i*2][j*2+1], mat[i*2+1][j*2], mat[i*2+1][j*2+1]))

    mat_new = np.array(mat_new)
    return mat_new

def bilinear(mat, width_new):
    # width_new 是32
    width = 16
    width_upsampling = width_new
    mat_upsampling = np.zeros((width_upsampling, width_upsampling))
    for i in range(width_upsampling):
        for j in range(width_upsampling):
            src_x = j * float(width / width_upsampling)
            src_y = i * float(width / width_upsampling)
            src_x_int = j * width // width_upsampling
            src_y_int = i * width // width_upsampling
            a = src_x - src_x_int
            b = src_y - src_y_int

            if src_x_int+1 == width or src_y_int+1 == width:
                mat_upsampling[i, j] = mat[src_y_int][src_x_int]
                continue
            # print(src_x_int, src_y_int)
            mat_upsampling[i, j] = (1. - a) * (1. - b) * mat[src_y_int+1][src_x_int+1] + \
                            (1. - a) * b * mat[src_y_int][src_x_int+1] + \
                            a * (1. - b) * mat[src_y_int+1][src_x_int] + \
                            a * b * mat[src_y_int][src_x_int]
    return mat_upsampling

if __name__ == '__main__':
    img_path = './1.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_shape = (img.shape[0], img.shape[1])
    dst_shape = (2*img_shape[0], 2*img_shape[1])
    dst_img = bilinear(img, img_shape, dst_shape)
    cv2.imwrite('./1_bilinear.jpg', dst_img)



