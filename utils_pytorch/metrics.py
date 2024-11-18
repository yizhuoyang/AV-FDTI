import time
from random import random

import cv2
import numpy as np
import torch


def calculate_result(result,o,z):
    # max_value = np.max(result)
    # max_index = np.unravel_index(np.argmax(result), result.shape)
    x_size, y_size = 512, 512
    x_len = 5.861056694237837 + 1.827702940235904
    y_len = 18.766594286932648 + 4.720740577048357
    x_down = 1.827702940235904
    y_down = 4.720740577048357
    ratiox = x_size / x_len
    ratioy = y_size / y_len
    sum_1 = np.sum(result, 0)
    sum_2 = np.sum(result, 1)
    rx, ry = np.argmax(sum_1), np.argmax(sum_2)
    # if rx<=1 or ry <=1 or rx>=510 or rx>=510:
    #     max_value = np.max(result)
    #     rx,ry = np.unravel_index(np.argmax(result), result.shape)
    px = rx / ratiox - x_down + o[0] / 100
    py = ry / ratioy - y_down + o[1] / 100
    result = np.array([px, py, z[0]])

    return result


