import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import random


def keypoint_detection(label, prominence=0.1):
    '''
    1. smooth the label
    2. Find the local maximum and minimum
    3. Denote the local maximmum to 1 and local minimum to -1, other to 0
    '''
    smoothed_label = label
    # smoothed_label = np.convolve(label, np.ones(5)/5, mode='same')
    peak = find_peaks(smoothed_label, distance=5, prominence=prominence)[0]
    valley = find_peaks(-smoothed_label, distance=5, prominence=prominence)[0]

    keypoint = np.zeros_like(label)
    keypoint[peak] = 1
    keypoint[valley] = -1
    # print(f"keypoint: {keypoint}")
    return keypoint

def select_keyframe(keyframe, start_idx, target_length, target_fps, fps_ori):
    # print(keyframe, start_idx, target_length, target_fps, fps_ori)

    # 1. Random select 6 keyframes where the keypoint is 1
    # 2. Random select 1 keyframes where keypoint is -1 between each two keyframes
    # 3. Random select until the target length is reached for keypoint 0

    end_idx = start_idx + round(target_length / target_fps * fps_ori)
    keyframe = keyframe[start_idx:end_idx]

    # print(f"start_idx: {start_idx}, end_idx: {end_idx}, target_length: {target_length}, target_fps: {target_fps}, fps_ori: {fps_ori}")
    
    pos_1 = np.where(keyframe == 1)[0]
    pos_neg_1 = np.where(keyframe == -1)[0]

    if len(pos_1) == 0:
        random_keypoint = [0, end_idx - start_idx - round(fps_ori / target_fps)]
    else:
        select_num = min(6, len(pos_1))
        random_peak = np.random.choice(pos_1, select_num, replace=False)
        random_valley = np.array([])
        if 0 not in random_peak:
            random_peak = np.insert(random_peak, 0, 0)
        else:
            select_num -= 1
        random_peak = np.sort(random_peak)
        for i in range(0, select_num):
            peak_1, peak_2 = random_peak[i], random_peak[i+1]

            all_valley = pos_neg_1[(pos_neg_1 > peak_1) & (pos_neg_1 < peak_2)]
            if len(all_valley) > 0:
                random_v= np.random.choice(all_valley, 1)
                random_valley = np.concatenate([random_valley, random_v])
        
        random_keypoint = np.sort(np.concatenate([random_peak, random_valley]))
    
    if len(random_keypoint) < target_length:
        idx_to_add = np.array([])
        num_padding = target_length - len(random_keypoint)

        start, intervel_len = random_keypoint[:-1], [random_keypoint[i+1] - random_keypoint[i] for i in range(len(random_keypoint)-1)]

        # num_padding_interval = intervel_len * num_padding / sum(intervel_len)
        num_padding_interval = [num_padding * l / sum(intervel_len) for l in intervel_len]
        
        for i in range(len(intervel_len)):
            if i == len(intervel_len) - 1 and num_padding > 0:
                idx_to_add = np.concatenate([idx_to_add, np.linspace(random_keypoint[i], random_keypoint[i+1], num_padding+1, endpoint=False).astype(int)[1:]])
            else:

                weight = intervel_len[i] / sum(intervel_len)

                padd_num = min(num_padding, int(weight // 1) + int(random.random() < weight % 1) )
                if padd_num > 0:
                    num_padding -= padd_num
                    
                    idx_to_add = np.concatenate([idx_to_add, np.linspace(random_keypoint[i], random_keypoint[i+1], padd_num+1, endpoint=False).astype(int)[1:]])
        
        random_keypoint = np.sort(np.concatenate([random_keypoint, idx_to_add]))
    random_keypoint = np.unique(random_keypoint)
    if len(random_keypoint) > target_length:
        random_keypoint = random_keypoint[:target_length]
    elif len(random_keypoint) < target_length:
        unselected = set(range(end_idx - start_idx)) - set(random_keypoint)
        extra = np.random.choice(list(unselected), target_length - len(random_keypoint), replace=False)
        random_keypoint = np.sort(np.concatenate([random_keypoint, extra]))
    frame_stride = random_keypoint[1:] - random_keypoint[:-1]
    frame_stride = np.insert(frame_stride, 0, round(fps_ori / target_fps))

    
    return random_keypoint.astype(int), frame_stride