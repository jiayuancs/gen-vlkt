"""
加载 results.pkl 和 results_ood.pkl
评估 OOD 性能
"""

import json
import ast
from code import interact
from fileinput import filename
from locale import normalize
import os
import torch
import pickle
import numpy as np
import scipy.io as sio
import json
from sklearn import metrics
from sklearn.metrics import roc_curve as Roc
from scipy import interpolate
from tqdm import tqdm


# 特定于项目
import sys
sys.path.append('/workspace/code/OpenSourceModel/pocket')
from pocket.utils import BoxPairAssociation

## OOD eval tools start

def _cal_auc_fpr(id_ness, labels):
    auroc = metrics.roc_auc_score(labels, id_ness)
    fpr,tpr,thresh = Roc(labels, id_ness, pos_label=1)
    fpr = float(interpolate.interp1d(tpr, fpr)(0.95))
    return auroc, fpr

to_np = lambda x: x.detach().cpu().numpy()
def max_logit_score(logits):
    return to_np(torch.max(logits, -1)[0])
def msp_score(logits):
    prob = torch.softmax(logits, -1)
    return to_np(torch.max(prob, -1)[0])
def energy_score(logits):
    return to_np(torch.logsumexp(logits, -1))

def merge_ood_results(ood_results_lh, ood_results_rh):
    """合并两个 OOD 任务输出的结果"""
    all_results = {}
    assert ood_results_lh.keys() == ood_results_rh.keys()
    for key in ood_results_lh.keys():
        lh_res = ood_results_lh[key]
        rh_res = ood_results_rh[key]
        all_results[key] = np.concatenate((lh_res, rh_res), axis=0)
    return all_results

def evaluate_ood_results(ood_results):
    """评估 OOD 任务输出的结果"""
    # ground-truth
    label_key_name = "label"
    labels = ood_results[label_key_name]  # [n, 1]

    eval_results = {}
    for key, value in ood_results.items():
        if key == label_key_name:
            continue
        auroc, fpr = _cal_auc_fpr(id_ness=value, labels=labels)
        eval_results[key] = (auroc, fpr)
    
    return eval_results



def eval_ood(data):
    """
    data: 模型输出的预测结果，该文件是一个 dict，具有如下两个字段：
              - "label": 形状为 [N] 的 numpy 数组, 表示 ground-truth, 1 表示 ID, 0 表示 OOD
              - "logit": 形状为 [N, 117] 的 numpy 数组, 表示模型输出的动作类别置信度分数
              - "atd": 形状为 [N, 1] 的 numpy 数组, 可选
              - "ctw": 形状为 [N, 1] 的 numpy 数组, 可选
    """
    label = data["label"]
    logit = torch.from_numpy(data["logit"])

    # ID 样本和 OOD 样本的比例
    pos_cnt = np.count_nonzero(label)
    neg_cnt = label.shape[0] - pos_cnt
    print(f"ID/OOD: {pos_cnt}/{neg_cnt}")

    ind_logits = max_logit_score(logit)
    ind_prob = msp_score(logit)
    ind_energy = energy_score(logit)

    ood_results = {
        "label": label,
        "MSP": ind_prob,
        "MaxLogit": ind_logits,
        "Energy": ind_energy
    }

    if "atd" in data.keys():
        ood_results["ATD"] = data["atd"]
    if "ctw" in data.keys():
        ood_results["CTW"] = data["ctw"]

    ood_performance = evaluate_ood_results(ood_results)
    for metric_name, (auroc, fpr) in ood_performance.items():
        print(f"{metric_name}: auroc={auroc*100:.2f}, fpr={fpr*100:.2f}")
    return ood_performance

## OOD eval tools end


def load_pkl_file(filepath):
    try:
        print(f"loading {filepath}")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception:
        print(f"FAILED to load json file: {filepath}")


def match_with_iou(data_dict, id_flag):
    all_preds = data_dict["preds"]
    all_ood_preds = data_dict["ood_preds"]
    all_gts = data_dict["gts"]
    assert len(all_preds) == len(all_gts)

    associate = BoxPairAssociation(min_iou=0.5)

    all_label = []
    all_logit = []
    for preds, ood_preds, gts in tqdm(zip(all_preds, all_ood_preds, all_gts), total=len(all_preds)):
        assert preds["filename"] == gts["filename"] and ood_preds["filename"] == gts["filename"]
        
        ###### 模型输出 ########
        boxes = torch.tensor([box['bbox'] for box in preds["predictions"]])
        hbox_idx = torch.tensor(ood_preds["subject_id"])
        obox_idx = torch.tensor(ood_preds["object_id"])
        score = torch.tensor(ood_preds["score"])
        object_class = torch.tensor(ood_preds["object_class"])

        ###### ground-truth ########
        gt_boxes = torch.tensor([box['bbox'] for box in gts["annotations"]])
        gt_boxes_label = torch.tensor([box['category_id'] for box in gts["annotations"]])
        gt_boxes_h_idx = torch.tensor([tri['subject_id'] for tri in gts['hoi_annotation']])
        gt_boxes_o_idx = torch.tensor([tri['object_id'] for tri in gts['hoi_annotation']])

        # 构造用于评测
        all_scores = score
        ood_boxes_h = boxes[hbox_idx]
        ood_boxes_o = boxes[obox_idx]
        all_objects = object_class[obox_idx]

        # gound-truth
        gt_bx_h = gt_boxes[gt_boxes_h_idx]
        gt_bx_o = gt_boxes[gt_boxes_o_idx]
        gt_object_class = gt_boxes_label[gt_boxes_o_idx]

        # 匹配边界框，得到 ground-truth 标签(1 表示 ID 人物对，0 表示 OOD 人物对)
        # target['object']
        # all_objects
        unique_object = all_objects.unique()
        for obj_idx in unique_object:
            gt_idx = torch.nonzero(gt_object_class == obj_idx).squeeze(1)
            det_idx = torch.nonzero(all_objects == obj_idx).squeeze(1)
            if len(gt_idx):
                ood_label = associate(
                    (gt_bx_h[gt_idx].view(-1, 4),
                    gt_bx_o[gt_idx].view(-1, 4)),
                    (ood_boxes_h[det_idx].view(-1, 4),
                    ood_boxes_o[det_idx].view(-1, 4)),
                    None   # 对于重复匹配的人物对，仅保留 IoU 最大的人物对
                )
                # 仅保留与 ground-truth 匹配的 人物对
                d2idxs = torch.nonzero(ood_label, as_tuple=False)
                idxs = det_idx[d2idxs]
                pos_score = all_scores[idxs].squeeze(1)

                # 匹配的人物对
                if id_flag:
                    all_label.append(torch.ones_like(idxs))
                else:
                    all_label.append(torch.zeros_like(idxs))
                all_logit.append(pos_score)

    match_ood_results = {
        "label": torch.cat(all_label).squeeze(-1).numpy(),
        "logit": torch.cat(all_logit).numpy()
    }

    return match_ood_results


def process_data(id_data, ood_data):
    match_ood_results_id_part = match_with_iou(id_data, id_flag=True)
    match_ood_results_ood_part = match_with_iou(ood_data, id_flag=False)
    all_match_ood_results = merge_ood_results(ood_results_lh=match_ood_results_id_part, ood_results_rh=match_ood_results_ood_part)
    ood_performance = eval_ood(all_match_ood_results)


if __name__ == "__main__":
    id_data = load_pkl_file("results.pkl")
    ood_data = load_pkl_file("results_ood.pkl")

    process_data(id_data, ood_data)
    print("Done")
