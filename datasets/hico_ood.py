"""
HICO detection dataset.

SWIG-OOD dataset

just for OOD Detection
"""
from pathlib import Path

import torchvision.transforms
from PIL import Image
import json
from collections import defaultdict
import numpy as np

import os
import torch
import torch.utils.data
import clip

import datasets.transforms as T
from .hico_text_label import hico_text_label, hico_unseen_index



class OODDetection(torch.utils.data.Dataset):
    def __init__(self, img_set, img_folder, anno_file, clip_feats_folder, transforms, num_queries, args):
        self.img_set = img_set
        self.img_folder = img_folder
        self.clip_feates_folder = clip_feats_folder
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        self._transforms = transforms

        self.num_queries = num_queries
        self.ood_dataset_flag = True

        self.rare_triplets = []
        self.non_rare_triplets = []
        self.correct_mat = []
        # self.unseen_index = hico_unseen_index.get(args.zero_shot_type, [])
        # self.unseen_index = []
        # self._valid_obj_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
        #                        14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        #                        24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
        #                        37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
        #                        48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
        #                        58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
        #                        72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
        #                        82, 84, 85, 86, 87, 88, 89, 90)
        # self._valid_verb_ids = list(range(1, 118))

        # self.text_label_dict = hico_text_label
        # self.text_label_ids = list(self.text_label_dict.keys())
        if img_set == 'train' and len(self.unseen_index) != 0 and args.del_unseen:
            raise NotImplementedError
            # tmp = []
            # for idx, k in enumerate(self.text_label_ids):
            #     if idx in self.unseen_index:
            #         continue
            #     else:
            #         tmp.append(k)
            # self.text_label_ids = tmp

        if img_set == 'train':
            raise NotImplementedError
            # self.ids = []
            # for idx, img_anno in enumerate(self.annotations):
            #     new_img_anno = []
            #     skip_pair = []
            #     for hoi in img_anno['hoi_annotation']:
            #         if hoi['hoi_category_id'] - 1 in self.unseen_index:
            #             skip_pair.append((hoi['subject_id'], hoi['object_id']))
            #     for hoi in img_anno['hoi_annotation']:
            #         if hoi['subject_id'] >= len(img_anno['annotations']) or hoi['object_id'] >= len(
            #                 img_anno['annotations']):
            #             new_img_anno = []
            #             break
            #         if (hoi['subject_id'], hoi['object_id']) not in skip_pair:
            #             new_img_anno.append(hoi)
            #     if len(new_img_anno) > 0:
            #         self.ids.append(idx)
            #         img_anno['hoi_annotation'] = new_img_anno
        else:
            self.ids = list(range(len(self.annotations['filenames'])))
        print("{} contains {} images".format("OOD", len(self.ids)))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # _, self.clip_preprocess = clip.load(args.clip_model, device)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_anno = self.annotations['annotation'][self.ids[idx]]
        img_name = self.annotations['filenames'][self.ids[idx]]
        ho_cnt = len(img_anno['boxes_h'])

        img = Image.open(os.path.join(self.img_folder, img_name)).convert('RGB')
        w, h = img.size

        if self.img_set == 'train' and len(img_anno['annotations']) > self.num_queries:
            raise NotImplementedError
            # img_anno['annotations'] = img_anno['annotations'][:self.num_queries]

        # boxes = [obj['bbox'] for obj in img_anno['annotations']]
        # 格式为[x1, y1, x2, y2], 原始图像尺寸
        boxes = img_anno['boxes_h'] + img_anno['boxes_o']
        # # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        if self.img_set == 'train':
            raise NotImplementedError
            # Add index for confirming which boxes are kept after image transformation
            # classes = [(i, self._valid_obj_ids.index(obj['category_id'])) for i, obj in
            #            enumerate(img_anno['annotations'])]
        else:
            # classes = [self._valid_obj_ids.index(obj['category_id']) for obj in img_anno['annotations']]
            classes = [0] * ho_cnt + img_anno['object']
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['size'] = torch.as_tensor([int(h), int(w)])
        if self.img_set == 'train':
            raise NotImplementedError
        else:
            target['filename'] = img_name
            target['boxes'] = boxes
            target['labels'] = classes
            target['id'] = idx

            if self._transforms is not None:
                img, _ = self._transforms(img, None)



            hois = []
            for ho_idx in range(ho_cnt):
                hois.append((ho_idx, ho_idx+ho_cnt, img_anno['verb'][ho_idx]))
            # for hoi in img_anno['hoi_annotation']:
            #     hois.append((hoi['subject_id'], hoi['object_id'], self._valid_verb_ids.index(hoi['category_id'])))
            target['hois'] = torch.as_tensor(hois, dtype=torch.int64)


        return img, target

    def set_rare_hois(self, anno_file):
        raise NotImplementedError
        with open(anno_file, 'r') as f:
            annotations = json.load(f)

        if len(self.unseen_index) == 0:
            # no unseen categoruy, use rare to evaluate
            counts = defaultdict(lambda: 0)
            for img_anno in annotations:
                hois = img_anno['hoi_annotation']
                bboxes = img_anno['annotations']
                for hoi in hois:
                    triplet = (self._valid_obj_ids.index(bboxes[hoi['subject_id']]['category_id']),
                            self._valid_obj_ids.index(bboxes[hoi['object_id']]['category_id']),
                            self._valid_verb_ids.index(hoi['category_id']))
                    counts[triplet] += 1
            self.rare_triplets = []
            self.non_rare_triplets = []
            for triplet, count in counts.items():
                if count < 10:
                    self.rare_triplets.append(triplet)
                else:
                    self.non_rare_triplets.append(triplet)
            print("rare:{}, non-rare:{}".format(len(self.rare_triplets), len(self.non_rare_triplets)))
        else:
            self.rare_triplets = []
            self.non_rare_triplets = []
            for img_anno in annotations:
                hois = img_anno['hoi_annotation']
                bboxes = img_anno['annotations']
                for hoi in hois:
                    triplet = (self._valid_obj_ids.index(bboxes[hoi['subject_id']]['category_id']),
                            self._valid_obj_ids.index(bboxes[hoi['object_id']]['category_id']),
                            self._valid_verb_ids.index(hoi['category_id']))
                    if hoi['hoi_category_id'] - 1 in self.unseen_index:
                        self.rare_triplets.append(triplet)
                    else:
                        self.non_rare_triplets.append(triplet)
            print("unseen:{}, seen:{}".format(len(self.rare_triplets), len(self.non_rare_triplets)))

    def load_correct_mat(self, path):
        raise NotImplementedError
        self.correct_mat = np.load(path)


# Add color jitter to coco transforms
def make_hico_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return [T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(.4, .4, .4),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ]))]
            ),
            normalize
            ]

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_ood_dataset(image_set, args):
    root="/workspace/dataset/swig-hoi/images_512"
    anno_file="/workspace/dataset/swig-hoi/tmp/swig_hico.json"
    object_cls_num=80
    verb_cls_num=305
    hoi_cls_num=1161

    dataset = OODDetection(image_set, root, anno_file, None,
                            transforms=make_hico_transforms(image_set),
                            num_queries=args.num_queries, args=args)

    return dataset


if __name__ == "__main__":
    root="/workspace/dataset/swig-hoi/images_512"
    anno_file="/workspace/dataset/swig-hoi/tmp/swig_hico.json"
    image_set="val"
    dataset = OODDetection(image_set, root, anno_file, None,
                            transforms=make_hico_transforms(image_set),
                            num_queries=64, args=None)

    print(dataset[0])