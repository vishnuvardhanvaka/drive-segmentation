import os
import numpy as np
import random
import torch
import cv2


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def epoch_time(start_time,end_time):
    elapsed_time=end_time-start_time
    mins=int(elapsed_time/60)
    return mins


















