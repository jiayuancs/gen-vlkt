# gen-vlkt eval-ood 分支

运行如下指令生成 `results.pkl` 和 `results_ood.pkl` 文件：

```shell
python main.py \
        --pretrained pretrained/hico_gen_vlkt_s.pth \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers 3 \
        --eval \
        --with_clip_label \
        --with_obj_clip_label \
        --use_nms_filter
```

运行 `eval_ood.py` 评估 `results.pkl` 和 `results_ood.pkl` 文件，输出 OOD 检测性能：

```shell
python eval_ood.py
```
