import json
import sys
import os
import numpy as np


def calculate_class_motion_score(motion='/dockerx/local/data/AVSync15/test_curves_npy'):

    result  = {}
    for cate in os.listdir(motion):
        print(cate)
        for video in os.listdir(os.path.join(motion, cate)):
            motion_path = os.path.join(motion, cate, video)
            motion_s = np.load(motion_path)

            m = motion_s.max()
            if cate not in result:
                result[cate] = m
            else:
                result[cate] += m
        result[cate] /= len(os.listdir(os.path.join(motion, cate)))
    result = sorted(result.items(), key=lambda x: x[1])
    print(result)
    return

def summarize_overall(results_path):

    with open(results_path, "r") as f:
        data = json.load(f)


    instances_score = data['instance_metrics']

    categories_score = {}
    low_score = {}
    lower_bound = 0.4

    for instance in instances_score:
        instance = instance
        score = instances_score[instance]["RelSync"]
        cate, name = instance.split("/")

        if cate not in categories_score:
            categories_score[cate] = []
        categories_score[cate].append(score)

        if score < lower_bound:
            low_score[instance] = score

    outpout = "Category,Mean,Min,Max,Low\n"
    for cate in categories_score:
        scores = categories_score[cate]
        outpout += "{},{},{},{},{}\n".format(cate, sum(scores) / len(scores), min(scores), max(scores), len([s for s in scores if s < lower_bound]))
    save_path = results_path.replace(".json", "_summary.csv")
    with open(save_path, "w") as f:
        f.write(outpout)
    print("Save to: ", save_path)

def find_worst_instance_to_baseline(results_path, asva_path):
    with open(results_path, "r") as f:
        result = json.load(f)['instance_metrics']
    with open(asva_path, "r") as f:
        asva = json.load(f)['instance_metrics']
    
    threshold = 0.2
    worse_list = []
    for instance in result:
        if instance not in asva:
            print("Instance not found in ASVA: ", instance)
        else:
            worse_list.append((instance, result[instance]["RelSync"] - asva[instance]["RelSync"], 
                                    result[instance]["RelSync"], asva[instance]["RelSync"]))
                # print("Worse than ASVA: ", instance, result[instance]["RelSync"], asva[instance]["RelSync"])
    worse_list = sorted(worse_list, key=lambda x: x[1])
    # save the result
    save_path = results_path.replace(".json", "_worse.csv")
    with open(save_path, "w") as f:
        f.write("Instance, Compare, Qformer, ASVA\n")
        for instance in worse_list:
            f.write("{},{},{}, {}\n".format(instance[0], instance[1], instance[2], instance[3]))


def plot_cate_result(exp_list, name):

    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.cm as cm

    fig, ax = plt.subplots()
    # figure size
    fig.set_size_inches(28.5, 20.5)
    W = 2
    width = W / (len(exp_list) + 1)
    x = np.arange(15) * W
    colors = cm.get_cmap('viridis', len(exp_list))

    for i, exp in enumerate(exp_list):
        cate = []
        mean = []
        min_l = []
        max_l = []
        low = []
        with open(exp, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                cate_, mean_, min_, max_, low_ = line.strip().split(",")
                cate.append(cate_)
                mean.append(float(mean_))
                min_l.append(float(min_))
                max_l.append(float(max_))
                low.append(float(low_))
        
        ax.bar(x + i * width, mean, width, label=name[i], color=colors(i))
    
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(cate, fontsize=15)
    ax.set_ylim(0.35, 0.6)
    plt.xticks(rotation=45)
    ax.legend()
    
    plt.subplots_adjust(bottom=0.3)  
    plt.savefig("cate_result.png")






if __name__ == "__main__":
    # ASVA baseline
    # asva_path = "/dockerx/local/backup_2/DynamiCrafter/save/training_512_avsyn_n/metrics/eval_result.json"
    
    # # Original 12frames Qformer
    # # results_path = "/dockerx/share/DynamiCrafter/save/training_512_avsyn_qformer_12_keyframe_framequery_uniform12-489_train_set/metrics/eval_result_asva.json"
    # results_path = "/dockerx/share/Dynamicrafter_audio/save/asva/asva_12_freenoise/metrics/epoch=1419-step=34080_audio_11.0_img_2.0-uniform.json"

    # results_path_2 = '/dockerx/share/Dynamicrafter_audio/save/asva/asva_12_freenoise/metrics/epoch=1419-step=34080_audio_11.0_img_2.0.json'
    # results_path_3 = '/dockerx/share/Dynamicrafter_audio/save/asva/asva_12_freenoise/metrics/tmp-12-more.json'

    # result_48_1 = '/dockerx/share/Dynamicrafter_audio/save/asva/asva_12_freenoise/metrics/tmp-10000-48-concat.json'
    # result_48_2 = '/dockerx/share/Dynamicrafter_audio/save/asva/asva_12_freenoise/metrics/tmp-10000-48-uniform.json'
    # # result_48_2 = '/dockerx/share/Dynamicrafter_audio/save/asva/asva_12_freenoise/metrics/tmp-10000-48-concat-no-index.json'

    result_kf = '/dockerx/share/Dynamicrafter_audio/save/asva/asva_12_freenoise/metrics/tmp-12-kf.json'
    # result_kf = '/dockerx/share/Dynamicrafter_audio/save/asva/asva_12_freenoise/metrics/tmp-12-kf-fix-baby.json'

    
    result_uniform='/dockerx/share/Dynamicrafter_audio/save/asva/asva_12_freenoise/metrics/tmp-12-uniform.json'
    

    summarize_overall(result_kf)
    summarize_overall(result_uniform)
    # find_worst_instance_to_baseline(result_48_1, result_48_2)

    # version_2_ckpt_1 = '/dockerx/share/DynamiCrafter/save/training_512_avsyn_qformer_12_keyframe_framequery_uniform12-489/metrics/eval_result_asva_summary.csv'
    # version_2_ckpt_2 = '/dockerx/share/DynamiCrafter/save/training_512_avsyn_qformer_12_keyframe_framequery_uniform12/metrics/eval_result_asva_summary.csv'
    # version_1 = '/dockerx/share/DynamiCrafter/save/training_512_avsyn_n/metrics/eval_result_2_qformer12_summary.csv'
    # baseline = '/dockerx/share/DynamiCrafter/save/training_512_avsyn_n/metrics/eval_result_asva_summary.csv'
    # plot_cate_result([version_2_ckpt_1, version_2_ckpt_2, version_1, baseline], name = ["v2_ckpt1_489_epoch", "v2_ckpt2", "v1", "baseline"])

    # version_1 = '/dockerx/share/Dynamicrafter_audio/save/asva/asva_12_freenoise/metrics/epoch=1419-step=34080_audio_11.0_img_2.0-uniform_summary.csv'
    # version_2 = '/dockerx/share/Dynamicrafter_audio/save/asva/asva_12_freenoise/metrics/epoch=1419-step=34080_audio_11.0_img_2.0_summary.csv'
    # version_3 = '/dockerx/share/Dynamicrafter_audio/save/asva/asva_12_freenoise/metrics/tmp-12-more_summary.csv'
    
    result_48_1 = '/dockerx/share/Dynamicrafter_audio/save/asva/asva_12_freenoise/metrics/tmp-12-kf-fix-baby_summary.csv'
    result_48_2 = '/dockerx/share/Dynamicrafter_audio/save/asva/asva_12_freenoise/metrics/tmp-12-uniform_summary.csv'
    plot_cate_result([result_48_2, result_48_1], ['Uniform', 'KF'])

    # calculate_class_motion_score()
