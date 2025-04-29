import os
import numpy as np
from numpy import dot
from numpy.linalg import norm



import matplotlib.pyplot as plt
def motion_sim(a, b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim


pred_ours = '/dockerx/share/Dynamicrafter_audio/save/vgg_sound/training_512_avsyn_qformer_12_keyframe_framequery_60audio_random_frames_fps_cf/epoch=3-step=19200/curves_npy'
pred_path='/dockerx/local/backup_2/ASVA/checkpoints/audio-cond_animation/avsync15_audio-cond_cfg/evaluations/checkpoint-37000/AG-4.0_TG-1.0/seed-0/curves_npy_test'
gt_path = '/dockerx/local/backup_2/ASVA/datasets/AVSync15/curves_npy'


score = {}
score_2 = {}
for cate in os.listdir(gt_path):
    score[cate] = []
    score_2[cate] = []
    for file in os.listdir(os.path.join(gt_path, cate)):
        gt = np.load(os.path.join(gt_path, cate, file))
        pred = np.load(os.path.join(pred_path, cate, file.replace('clip-', 'clip-0')))
        pred_2 = np.load(os.path.join(pred_ours, cate, file.replace('clip-', 'clip-0')))
        score[cate].append(motion_sim(gt, pred))
        score_2[cate].append(motion_sim(gt, pred_2))
        
        # plt.plot(gt, label='gt')
        # plt.plot(pred, label='pred')
        # plt.plot(pred_2, label='pred_ours')
        # plt.legend()
        # plt.savefig('temp.png')
        # plt.close()
        # import ipdb; ipdb.set_trace()
        


for cate in score:
    print(cate, np.mean(score[cate]), np.mean(score_2[cate]))
    # print(cate, np.mean(score_2[cate]))
# baby_babbling_crying/5Uodeo_Ln84_000014_000024_6.5_9.5_clip-00.mp4.npy
# baby_babbling_crying/5Uodeo_Ln84_000014_000024_6.5_9.5_clip-0.mp4.npy