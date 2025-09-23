#!/usr/bin/env python3
"""
Created on 2025-09-23 (Tue) 00:01:52

Code for ADD-GCN (https://github.com/Yejin0111/ADD-GCN) original run.

### Modifications
- Remove unzip process in coco.py (because I have already unzipped the files)
- Change num_workers from 4 to 1 (to avoid OOM error)

@author: I.Azuma
"""
# %%
PROJECT_DIR = "/workspace/cluster/HDD/azuma/Others/github/ADD-GCN"
DATA_DIR = "/workspace/cluster/HDD/azuma/Others/datasource"
SAVE_DIR = "/workspace/cluster/HDD/azuma/Others/github/ML_DL_Notebook/MultiLabel_Image_Recognition/ADD-GCN/results"

import os
os.chdir(PROJECT_DIR)

! python main.py --data COCO2014 --data_root_dir {DATA_DIR} --model_name ADD_GCN --save_dir {SAVE_DIR} -b 16 --lr 0.05  --num_workers 1

"""
* absolute seed: 1
/workspace/cluster/HDD/azuma/Others/github/ADD-GCN/main.py:48: UserWarning: You have chosen to seed training. This will turn on the CUDNN deterministic setting, which can slow down your training considerably! You may see unexpected behavior when restarting from checkpoints.
  warnings.warn('You have chosen to seed training. '
[json] Done!
[dataset] COCO2014 classification phase=val number of classes=80  number of images=40137
[json] Done!
[dataset] COCO2014 classification phase=train number of classes=80  number of images=82081
/opt/250102_test_env/lib/python3.11/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/opt/250102_test_env/lib/python3.11/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet101_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet101_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
--------Args Items----------
data_root_dir: /workspace/cluster/HDD/azuma/Others/datasource
image_size: 448
epochs: 50
epoch_step: [30, 40]
batch_size: 16
num_workers: 1
display_interval: 200
lr: 0.05
lrp: 0.1
momentum: 0.9
weight_decay: 0.0001
max_clip_grad_norm: 10.0
seed: 1
data: COCO2014
model_name: ADD_GCN
save_dir: /workspace/cluster/HDD/azuma/Others/github/ML_DL_Notebook/MultiLabel_Image_Recognition/ADD-GCN/results
evaluate: False
resume: 
--------Args Items----------

/workspace/cluster/HDD/azuma/Others/github/ADD-GCN/util.py:58: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  self.scores = torch.FloatTensor(torch.FloatStorage())
Lr: [0.005 0.05 ]
2025-09-23 00:45:35, 1 Epoch, 0 Iter, Loss: 52.3175, Data time: 3.3455, Batch time: 6.8197
2025-09-23 00:47:23, 1 Epoch, 200 Iter, Loss: 0.1351, Data time: 0.7132, Batch time: 0.9270
2025-09-23 00:49:19, 1 Epoch, 400 Iter, Loss: 0.1193, Data time: 0.3900, Batch time: 0.5001
2025-09-23 00:51:13, 1 Epoch, 600 Iter, Loss: 0.0979, Data time: 0.2947, Batch time: 0.3997
2025-09-23 00:53:00, 1 Epoch, 800 Iter, Loss: 0.1091, Data time: 0.3329, Batch time: 0.4417
2025-09-23 00:54:47, 1 Epoch, 1000 Iter, Loss: 0.1124, Data time: 0.4176, Batch time: 0.5385
2025-09-23 00:56:37, 1 Epoch, 1200 Iter, Loss: 0.1136, Data time: 0.3874, Batch time: 0.5705
2025-09-23 00:58:28, 1 Epoch, 1400 Iter, Loss: 0.0862, Data time: 0.3319, Batch time: 0.4828
2025-09-23 01:00:17, 1 Epoch, 1600 Iter, Loss: 0.1258, Data time: 0.4842, Batch time: 0.5999
2025-09-23 01:02:10, 1 Epoch, 1800 Iter, Loss: 0.1128, Data time: 0.3435, Batch time: 0.5245
2025-09-23 01:04:04, 1 Epoch, 2000 Iter, Loss: 0.1042, Data time: 0.2749, Batch time: 0.4726
2025-09-23 01:05:56, 1 Epoch, 2200 Iter, Loss: 0.0865, Data time: 0.2816, Batch time: 0.4021
2025-09-23 01:07:39, 1 Epoch, 2400 Iter, Loss: 0.1049, Data time: 0.2736, Batch time: 0.3992
2025-09-23 01:09:24, 1 Epoch, 2600 Iter, Loss: 0.1074, Data time: 0.2691, Batch time: 0.3958
2025-09-23 01:11:10, 1 Epoch, 2800 Iter, Loss: 0.0686, Data time: 0.1834, Batch time: 0.3016
2025-09-23 01:12:58, 1 Epoch, 3000 Iter, Loss: 0.0665, Data time: 0.2877, Batch time: 0.3995
2025-09-23 01:14:47, 1 Epoch, 3200 Iter, Loss: 0.0848, Data time: 0.3990, Batch time: 0.5053
2025-09-23 01:16:38, 1 Epoch, 3400 Iter, Loss: 0.0793, Data time: 0.2259, Batch time: 0.3981
2025-09-23 01:18:31, 1 Epoch, 3600 Iter, Loss: 0.0940, Data time: 0.4307, Batch time: 0.6167
2025-09-23 01:20:25, 1 Epoch, 3800 Iter, Loss: 0.0617, Data time: 0.3058, Batch time: 0.4102
2025-09-23 01:22:19, 1 Epoch, 4000 Iter, Loss: 0.0811, Data time: 0.2880, Batch time: 0.4026
2025-09-23 01:24:15, 1 Epoch, 4200 Iter, Loss: 0.0439, Data time: 0.2476, Batch time: 0.4292
2025-09-23 01:26:11, 1 Epoch, 4400 Iter, Loss: 0.0624, Data time: 0.3105, Batch time: 0.4117
2025-09-23 01:28:07, 1 Epoch, 4600 Iter, Loss: 0.0421, Data time: 0.3081, Batch time: 0.4145
2025-09-23 01:30:04, 1 Epoch, 4800 Iter, Loss: 0.0649, Data time: 0.3148, Batch time: 0.4204
2025-09-23 01:32:02, 1 Epoch, 5000 Iter, Loss: 0.0447, Data time: 0.3144, Batch time: 0.4822
Validate: 100%|█████████████████████████████| 2509/2509 [18:44<00:00,  2.23it/s]
tensor([0.9495, 0.4403, 0.3945, 0.6712, 0.8009, 0.8657, 0.9324, 0.7234, 0.4828,
        0.6770, 0.6820, 0.7847, 0.5273, 0.5275, 0.5586, 0.8463, 0.8144, 0.5936,
        0.8146, 0.3392, 0.9395, 0.4358, 0.6328, 0.7639, 0.6317, 0.8055, 0.6083,
        0.7337, 0.7588, 0.6097, 0.9667, 0.6149, 0.4819, 0.6103, 0.9851, 0.0222,
        0.4295, 0.8649, 0.4751, 0.7942, 0.8815, 0.4206, 0.7286, 0.5042, 0.8909,
        0.7524, 0.7000, 0.7353, 0.4291, 0.9846, 0.9146, 0.5596, 0.5157, 0.5260,
        0.4760, 0.2832, 0.9038, 0.7878, 0.8940, 0.8946, 0.4716, 0.4053, 0.7629,
        0.6357, 0.4166, 0.8610, 0.7720, 0.9748, 0.7567, 0.0979, 0.9561, 0.3510,
        0.7249, 0.9365, 0.6204, 0.7688, 0.8099, 0.6591, 0.5265, 0.9784])
* Test
Loss: 0.0583	 mAP: 0.6732	Data_time: 0.3524	 Batch_time: 0.4479
OP: 0.881	 OR: 0.518	 OF1: 0.653	CP: 0.824	 CR: 0.428	 CF1: 0.563
OP_3: 0.901	 OR_3: 0.491	 OF1_3: 0.636	CP_3: 0.837	 CR_3: 0.403	 CF1_3: 0.544
Save checkpoint to /workspace/cluster/HDD/azuma/Others/github/ML_DL_Notebook/MultiLabel_Image_Recognition/ADD-GCN/results/checkpoint.pth
Save results to /workspace/cluster/HDD/azuma/Others/github/ML_DL_Notebook/MultiLabel_Image_Recognition/ADD-GCN/results/results.csv
 * best mAP=0.6732
Lr: [0.005 0.05 ]
2025-09-23 01:53:44, 2 Epoch, 0 Iter, Loss: 0.0702, Data time: 0.7709, Batch time: 1.0449
2025-09-23 01:55:00, 2 Epoch, 200 Iter, Loss: 0.0688, Data time: 0.0250, Batch time: 0.1311
2025-09-23 01:56:15, 2 Epoch, 400 Iter, Loss: 0.0888, Data time: 0.0313, Batch time: 0.2049
2025-09-23 01:57:31, 2 Epoch, 600 Iter, Loss: 0.0595, Data time: 0.0337, Batch time: 0.3012
2025-09-23 01:58:48, 2 Epoch, 800 Iter, Loss: 0.0729, Data time: 0.0198, Batch time: 0.1349
2025-09-23 02:00:04, 2 Epoch, 1000 Iter, Loss: 0.0713, Data time: 0.0275, Batch time: 0.1385
2025-09-23 02:01:20, 2 Epoch, 1200 Iter, Loss: 0.0758, Data time: 0.2027, Batch time: 0.3216
2025-09-23 02:02:36, 2 Epoch, 1400 Iter, Loss: 0.0721, Data time: 0.0233, Batch time: 0.1399
2025-09-23 02:03:52, 2 Epoch, 1600 Iter, Loss: 0.0399, Data time: 0.0294, Batch time: 0.1333
2025-09-23 02:05:08, 2 Epoch, 1800 Iter, Loss: 0.0534, Data time: 0.1174, Batch time: 0.3206
2025-09-23 02:06:26, 2 Epoch, 2000 Iter, Loss: 0.0746, Data time: 0.0228, Batch time: 0.1296
2025-09-23 02:07:41, 2 Epoch, 2200 Iter, Loss: 0.0532, Data time: 0.1050, Batch time: 0.2840
2025-09-23 02:08:58, 2 Epoch, 2400 Iter, Loss: 0.0564, Data time: 0.1209, Batch time: 0.3282
2025-09-23 02:10:14, 2 Epoch, 2600 Iter, Loss: 0.0464, Data time: 0.1231, Batch time: 0.4037
2025-09-23 02:11:30, 2 Epoch, 2800 Iter, Loss: 0.0560, Data time: 0.0312, Batch time: 0.2223
2025-09-23 02:12:46, 2 Epoch, 3000 Iter, Loss: 0.0578, Data time: 0.1070, Batch time: 0.2322
2025-09-23 02:14:02, 2 Epoch, 3200 Iter, Loss: 0.0542, Data time: 0.0248, Batch time: 0.1302
2025-09-23 02:15:18, 2 Epoch, 3400 Iter, Loss: 0.0656, Data time: 0.0904, Batch time: 0.1993
2025-09-23 02:16:35, 2 Epoch, 3600 Iter, Loss: 0.0476, Data time: 0.1032, Batch time: 0.3031
2025-09-23 02:17:51, 2 Epoch, 3800 Iter, Loss: 0.0305, Data time: 0.1011, Batch time: 0.2060
2025-09-23 02:19:07, 2 Epoch, 4000 Iter, Loss: 0.0536, Data time: 0.1046, Batch time: 0.2936
2025-09-23 02:20:25, 2 Epoch, 4200 Iter, Loss: 0.0373, Data time: 0.0425, Batch time: 0.2266
2025-09-23 02:21:42, 2 Epoch, 4400 Iter, Loss: 0.0456, Data time: 0.0932, Batch time: 0.1890
2025-09-23 02:22:57, 2 Epoch, 4600 Iter, Loss: 0.0537, Data time: 0.1289, Batch time: 0.3172
2025-09-23 02:24:12, 2 Epoch, 4800 Iter, Loss: 0.0904, Data time: 0.1238, Batch time: 0.3176
2025-09-23 02:25:27, 2 Epoch, 5000 Iter, Loss: 0.0625, Data time: 0.0434, Batch time: 0.2172
Validate: 100%|█████████████████████████████| 2509/2509 [11:03<00:00,  3.78it/s]
tensor([0.9624, 0.6310, 0.4375, 0.8116, 0.8803, 0.8903, 0.9660, 0.8315, 0.5988,
        0.7429, 0.7738, 0.8449, 0.6091, 0.7037, 0.6872, 0.9160, 0.8549, 0.7638,
        0.8506, 0.5806, 0.9585, 0.5653, 0.7399, 0.7934, 0.7253, 0.8927, 0.7149,
        0.7704, 0.8611, 0.7885, 0.9793, 0.7541, 0.5799, 0.8658, 0.9949, 0.0871,
        0.4880, 0.9170, 0.7006, 0.8488, 0.9430, 0.5324, 0.8541, 0.6226, 0.9083,
        0.8017, 0.7724, 0.7501, 0.6357, 0.9884, 0.9284, 0.6345, 0.6671, 0.6917,
        0.6989, 0.4609, 0.9364, 0.8402, 0.9449, 0.9225, 0.7380, 0.4747, 0.8116,
        0.7235, 0.5885, 0.9126, 0.8452, 0.9845, 0.8072, 0.0888, 0.9674, 0.5835,
        0.7995, 0.9468, 0.6924, 0.8169, 0.8282, 0.7021, 0.6680, 0.9851])
* Test
Loss: 0.0482	 mAP: 0.7608	Data_time: 0.1519	 Batch_time: 0.2642
OP: 0.855	 OR: 0.663	 OF1: 0.747	CP: 0.822	 CR: 0.609	 CF1: 0.700
OP_3: 0.886	 OR_3: 0.606	 OF1_3: 0.720	CP_3: 0.844	 CR_3: 0.554	 CF1_3: 0.669
Save checkpoint to /workspace/cluster/HDD/azuma/Others/github/ML_DL_Notebook/MultiLabel_Image_Recognition/ADD-GCN/results/checkpoint.pth
Save results to /workspace/cluster/HDD/azuma/Others/github/ML_DL_Notebook/MultiLabel_Image_Recognition/ADD-GCN/results/results.csv
 * best mAP=0.7608
Lr: [0.005 0.05 ]
2025-09-23 02:39:03, 3 Epoch, 0 Iter, Loss: 0.0428, Data time: 0.7567, Batch time: 0.9433
2025-09-23 02:40:24, 3 Epoch, 200 Iter, Loss: 0.0522, Data time: 0.0893, Batch time: 0.1984
2025-09-23 02:41:46, 3 Epoch, 400 Iter, Loss: 0.0930, Data time: 0.1700, Batch time: 0.2995
2025-09-23 02:43:08, 3 Epoch, 600 Iter, Loss: 0.0529, Data time: 0.1248, Batch time: 0.4042
2025-09-23 02:44:29, 3 Epoch, 800 Iter, Loss: 0.0705, Data time: 0.1085, Batch time: 0.2847
2025-09-23 02:45:51, 3 Epoch, 1000 Iter, Loss: 0.0638, Data time: 0.1716, Batch time: 0.3013
2025-09-23 02:47:13, 3 Epoch, 1200 Iter, Loss: 0.0578, Data time: 0.0393, Batch time: 0.2297
2025-09-23 02:48:35, 3 Epoch, 1400 Iter, Loss: 0.0584, Data time: 0.0390, Batch time: 0.2199
2025-09-23 02:50:00, 3 Epoch, 1600 Iter, Loss: 0.0424, Data time: 0.1264, Batch time: 0.3158
2025-09-23 02:51:24, 3 Epoch, 1800 Iter, Loss: 0.0690, Data time: 0.1081, Batch time: 0.2998
2025-09-23 02:52:47, 3 Epoch, 2000 Iter, Loss: 0.0619, Data time: 0.1081, Batch time: 0.2854
2025-09-23 02:54:11, 3 Epoch, 2200 Iter, Loss: 0.0564, Data time: 0.1942, Batch time: 0.4717
2025-09-23 02:55:34, 3 Epoch, 2400 Iter, Loss: 0.0515, Data time: 0.1020, Batch time: 0.2300
2025-09-23 02:56:58, 3 Epoch, 2600 Iter, Loss: 0.0454, Data time: 0.1054, Batch time: 0.2317
2025-09-23 02:58:21, 3 Epoch, 2800 Iter, Loss: 0.0455, Data time: 0.0383, Batch time: 0.2106
2025-09-23 02:59:44, 3 Epoch, 3000 Iter, Loss: 0.0510, Data time: 0.1200, Batch time: 0.3003
2025-09-23 03:01:08, 3 Epoch, 3200 Iter, Loss: 0.0383, Data time: 0.1381, Batch time: 0.3224
2025-09-23 03:02:30, 3 Epoch, 3400 Iter, Loss: 0.0756, Data time: 0.0451, Batch time: 0.3038
2025-09-23 03:03:55, 3 Epoch, 3600 Iter, Loss: 0.0414, Data time: 0.1119, Batch time: 0.2999
2025-09-23 03:05:18, 3 Epoch, 3800 Iter, Loss: 0.0630, Data time: 0.1025, Batch time: 0.2291
2025-09-23 03:06:42, 3 Epoch, 4000 Iter, Loss: 0.0659, Data time: 0.1201, Batch time: 0.4151
2025-09-23 03:08:05, 3 Epoch, 4200 Iter, Loss: 0.0866, Data time: 0.1138, Batch time: 0.3239
2025-09-23 03:09:28, 3 Epoch, 4400 Iter, Loss: 0.0820, Data time: 0.1699, Batch time: 0.2882
2025-09-23 03:10:52, 3 Epoch, 4600 Iter, Loss: 0.0361, Data time: 0.1175, Batch time: 0.2352
2025-09-23 03:12:15, 3 Epoch, 4800 Iter, Loss: 0.0513, Data time: 0.0400, Batch time: 0.2174
2025-09-23 03:13:38, 3 Epoch, 5000 Iter, Loss: 0.0786, Data time: 0.1230, Batch time: 0.3323
Validate: 100%|█████████████████████████████| 2509/2509 [12:33<00:00,  3.33it/s]
tensor([0.9627, 0.6480, 0.4527, 0.8250, 0.9182, 0.9155, 0.9755, 0.8352, 0.6198,
        0.7732, 0.7816, 0.8678, 0.6471, 0.7105, 0.6983, 0.9228, 0.8609, 0.7886,
        0.8604, 0.7343, 0.9575, 0.6049, 0.7346, 0.8037, 0.7628, 0.9102, 0.7312,
        0.7696, 0.8781, 0.8100, 0.9809, 0.7882, 0.7354, 0.9050, 0.9931, 0.0662,
        0.5072, 0.9214, 0.7375, 0.8740, 0.9473, 0.6029, 0.8741, 0.6928, 0.9223,
        0.8450, 0.7892, 0.7990, 0.6508, 0.9893, 0.9333, 0.6585, 0.7143, 0.7197,
        0.7296, 0.4857, 0.9455, 0.8608, 0.9557, 0.9253, 0.7618, 0.5093, 0.8279,
        0.7281, 0.6612, 0.9292, 0.8646, 0.9871, 0.8205, 0.0924, 0.9711, 0.6139,
        0.8234, 0.9554, 0.7003, 0.8471, 0.8299, 0.7513, 0.7158, 0.9907])
* Test
Loss: 0.0461	 mAP: 0.7836	Data_time: 0.1866	 Batch_time: 0.3001
OP: 0.882	 OR: 0.660	 OF1: 0.755	CP: 0.850	 CR: 0.615	 CF1: 0.713
OP_3: 0.902	 OR_3: 0.611	 OF1_3: 0.728	CP_3: 0.869	 CR_3: 0.570	 CF1_3: 0.688
Save checkpoint to /workspace/cluster/HDD/azuma/Others/github/ML_DL_Notebook/MultiLabel_Image_Recognition/ADD-GCN/results/checkpoint.pth
Save results to /workspace/cluster/HDD/azuma/Others/github/ML_DL_Notebook/MultiLabel_Image_Recognition/ADD-GCN/results/results.csv
 * best mAP=0.7836
Lr: [0.005 0.05 ]
2025-09-23 03:28:48, 4 Epoch, 0 Iter, Loss: 0.0451, Data time: 0.8712, Batch time: 1.1724
2025-09-23 03:30:09, 4 Epoch, 200 Iter, Loss: 0.0339, Data time: 0.1050, Batch time: 0.2319
2025-09-23 03:31:31, 4 Epoch, 400 Iter, Loss: 0.0553, Data time: 0.0470, Batch time: 0.2297
2025-09-23 03:32:54, 4 Epoch, 600 Iter, Loss: 0.0583, Data time: 0.0897, Batch time: 0.1974
2025-09-23 03:34:18, 4 Epoch, 800 Iter, Loss: 0.0744, Data time: 0.3095, Batch time: 0.5147
2025-09-23 03:35:41, 4 Epoch, 1000 Iter, Loss: 0.0687, Data time: 0.0947, Batch time: 0.2058
2025-09-23 03:37:04, 4 Epoch, 1200 Iter, Loss: 0.0416, Data time: 0.0370, Batch time: 0.2056
2025-09-23 03:38:27, 4 Epoch, 1400 Iter, Loss: 0.0581, Data time: 0.0379, Batch time: 0.2172
2025-09-23 03:39:50, 4 Epoch, 1600 Iter, Loss: 0.0386, Data time: 0.1123, Batch time: 0.3010
2025-09-23 03:41:12, 4 Epoch, 1800 Iter, Loss: 0.0524, Data time: 0.1359, Batch time: 0.3300
2025-09-23 03:42:35, 4 Epoch, 2000 Iter, Loss: 0.0425, Data time: 0.1709, Batch time: 0.3016
2025-09-23 03:43:58, 4 Epoch, 2200 Iter, Loss: 0.0771, Data time: 0.0893, Batch time: 0.2011
2025-09-23 03:45:22, 4 Epoch, 2400 Iter, Loss: 0.0521, Data time: 0.1061, Batch time: 0.3032
2025-09-23 03:46:44, 4 Epoch, 2600 Iter, Loss: 0.0445, Data time: 0.1144, Batch time: 0.4117
2025-09-23 03:48:08, 4 Epoch, 2800 Iter, Loss: 0.0640, Data time: 0.0255, Batch time: 0.1311
2025-09-23 03:49:32, 4 Epoch, 3000 Iter, Loss: 0.0566, Data time: 0.1292, Batch time: 0.3042
2025-09-23 03:50:55, 4 Epoch, 3200 Iter, Loss: 0.0524, Data time: 0.1041, Batch time: 0.2352
2025-09-23 03:52:18, 4 Epoch, 3400 Iter, Loss: 0.0652, Data time: 0.1933, Batch time: 0.3708
2025-09-23 03:53:41, 4 Epoch, 3600 Iter, Loss: 0.0524, Data time: 0.2204, Batch time: 0.5068
2025-09-23 03:55:03, 4 Epoch, 3800 Iter, Loss: 0.0561, Data time: 0.1076, Batch time: 0.2339
2025-09-23 03:56:26, 4 Epoch, 4000 Iter, Loss: 0.0530, Data time: 0.1085, Batch time: 0.2321
2025-09-23 03:57:49, 4 Epoch, 4200 Iter, Loss: 0.0481, Data time: 0.1135, Batch time: 0.2360
2025-09-23 03:59:14, 4 Epoch, 4400 Iter, Loss: 0.0575, Data time: 0.1042, Batch time: 0.2327
2025-09-23 04:00:37, 4 Epoch, 4600 Iter, Loss: 0.0469, Data time: 0.1712, Batch time: 0.2987
2025-09-23 04:02:01, 4 Epoch, 4800 Iter, Loss: 0.0455, Data time: 0.1145, Batch time: 0.2334
2025-09-23 04:03:25, 4 Epoch, 5000 Iter, Loss: 0.0313, Data time: 0.1172, Batch time: 0.3321
Validate: 100%|█████████████████████████████| 2509/2509 [12:20<00:00,  3.39it/s]
tensor([0.9654, 0.6597, 0.4632, 0.8313, 0.9240, 0.9353, 0.9731, 0.8349, 0.6328,
        0.7519, 0.7993, 0.8818, 0.6546, 0.7220, 0.7126, 0.9296, 0.8734, 0.8024,
        0.8646, 0.7566, 0.9617, 0.6281, 0.7611, 0.8070, 0.7764, 0.9149, 0.7502,
        0.7811, 0.8772, 0.8148, 0.9798, 0.8001, 0.7522, 0.9172, 0.9944, 0.0535,
        0.5233, 0.9285, 0.7619, 0.8774, 0.9611, 0.6368, 0.8735, 0.7481, 0.9227,
        0.8583, 0.7933, 0.8043, 0.6622, 0.9892, 0.9370, 0.6702, 0.7154, 0.7465,
        0.7468, 0.5607, 0.9455, 0.8699, 0.9631, 0.9319, 0.7942, 0.5873, 0.8509,
        0.7531, 0.6830, 0.9363, 0.8660, 0.9872, 0.8295, 0.0917, 0.9725, 0.6622,
        0.8318, 0.9595, 0.7131, 0.8406, 0.8479, 0.7664, 0.7452, 0.9892])
* Test
Loss: 0.0445	 mAP: 0.7959	Data_time: 0.1816	 Batch_time: 0.2947
OP: 0.879	 OR: 0.680	 OF1: 0.767	CP: 0.849	 CR: 0.637	 CF1: 0.728
OP_3: 0.907	 OR_3: 0.619	 OF1_3: 0.736	CP_3: 0.872	 CR_3: 0.580	 CF1_3: 0.697
Save checkpoint to /workspace/cluster/HDD/azuma/Others/github/ML_DL_Notebook/MultiLabel_Image_Recognition/ADD-GCN/results/checkpoint.pth
Save results to /workspace/cluster/HDD/azuma/Others/github/ML_DL_Notebook/MultiLabel_Image_Recognition/ADD-GCN/results/results.csv
 * best mAP=0.7959
Lr: [0.005 0.05 ]
2025-09-23 04:18:22, 5 Epoch, 0 Iter, Loss: 0.0391, Data time: 0.9045, Batch time: 1.0872
2025-09-23 04:19:37, 5 Epoch, 200 Iter, Loss: 0.0433, Data time: 0.0285, Batch time: 0.1336
2025-09-23 04:20:53, 5 Epoch, 400 Iter, Loss: 0.0619, Data time: 0.0386, Batch time: 0.2273
2025-09-23 04:22:08, 5 Epoch, 600 Iter, Loss: 0.0551, Data time: 0.0330, Batch time: 0.2050
2025-09-23 04:23:24, 5 Epoch, 800 Iter, Loss: 0.0504, Data time: 0.0418, Batch time: 0.2279
2025-09-23 04:24:39, 5 Epoch, 1000 Iter, Loss: 0.0493, Data time: 0.0869, Batch time: 0.1974
2025-09-23 04:25:56, 5 Epoch, 1200 Iter, Loss: 0.0500, Data time: 0.0167, Batch time: 0.1325
2025-09-23 04:27:12, 5 Epoch, 1400 Iter, Loss: 0.0503, Data time: 0.0903, Batch time: 0.1975
2025-09-23 04:28:28, 5 Epoch, 1600 Iter, Loss: 0.0608, Data time: 0.1019, Batch time: 0.2810
2025-09-23 04:29:45, 5 Epoch, 1800 Iter, Loss: 0.0479, Data time: 0.0379, Batch time: 0.2298
2025-09-23 04:31:02, 5 Epoch, 2000 Iter, Loss: 0.0369, Data time: 0.0296, Batch time: 0.1348
2025-09-23 04:32:18, 5 Epoch, 2200 Iter, Loss: 0.0424, Data time: 0.0228, Batch time: 0.1296
2025-09-23 04:33:34, 5 Epoch, 2400 Iter, Loss: 0.0652, Data time: 0.0184, Batch time: 0.1293
2025-09-23 04:34:50, 5 Epoch, 2600 Iter, Loss: 0.0411, Data time: 0.0367, Batch time: 0.2236
2025-09-23 04:36:05, 5 Epoch, 2800 Iter, Loss: 0.0530, Data time: 0.1027, Batch time: 0.2789
2025-09-23 04:37:22, 5 Epoch, 3000 Iter, Loss: 0.0583, Data time: 0.0406, Batch time: 0.2274
2025-09-23 04:38:37, 5 Epoch, 3200 Iter, Loss: 0.0772, Data time: 0.0227, Batch time: 0.1283
2025-09-23 04:39:54, 5 Epoch, 3400 Iter, Loss: 0.0392, Data time: 0.0363, Batch time: 0.2216
2025-09-23 04:41:09, 5 Epoch, 3600 Iter, Loss: 0.0496, Data time: 0.1181, Batch time: 0.2358
2025-09-23 04:42:26, 5 Epoch, 3800 Iter, Loss: 0.0495, Data time: 0.0414, Batch time: 0.3002
2025-09-23 04:43:41, 5 Epoch, 4000 Iter, Loss: 0.0550, Data time: 0.0420, Batch time: 0.2275
2025-09-23 04:44:58, 5 Epoch, 4200 Iter, Loss: 0.0645, Data time: 0.0357, Batch time: 0.2228
2025-09-23 04:46:14, 5 Epoch, 4400 Iter, Loss: 0.0579, Data time: 0.0135, Batch time: 0.1299
2025-09-23 04:47:30, 5 Epoch, 4600 Iter, Loss: 0.0539, Data time: 0.0916, Batch time: 0.1978
2025-09-23 04:48:46, 5 Epoch, 4800 Iter, Loss: 0.0457, Data time: 0.0326, Batch time: 0.3004
2025-09-23 04:50:03, 5 Epoch, 5000 Iter, Loss: 0.0442, Data time: 0.0207, Batch time: 0.1295
Validate: 100%|█████████████████████████████| 2509/2509 [11:09<00:00,  3.75it/s]
tensor([0.9624, 0.6774, 0.4823, 0.8319, 0.9252, 0.9385, 0.9827, 0.8649, 0.6408,
        0.7756, 0.8005, 0.8825, 0.6676, 0.7268, 0.7253, 0.9340, 0.8745, 0.8189,
        0.8728, 0.7709, 0.9599, 0.6459, 0.7707, 0.8128, 0.7826, 0.9225, 0.7502,
        0.7838, 0.8904, 0.8392, 0.9801, 0.8302, 0.7688, 0.9286, 0.9942, 0.1397,
        0.5217, 0.9297, 0.7583, 0.8721, 0.9651, 0.6369, 0.8789, 0.7594, 0.9318,
        0.8510, 0.7982, 0.8117, 0.6521, 0.9901, 0.9422, 0.6819, 0.7430, 0.7220,
        0.7507, 0.5973, 0.9481, 0.8758, 0.9612, 0.9253, 0.8074, 0.5971, 0.8427,
        0.7640, 0.6882, 0.9434, 0.8621, 0.9853, 0.8348, 0.0988, 0.9745, 0.6641,
        0.8263, 0.9589, 0.7268, 0.8657, 0.8545, 0.7727, 0.7564, 0.9921])
* Test
Loss: 0.0439	 mAP: 0.8034	Data_time: 0.1536	 Batch_time: 0.2664
OP: 0.850	 OR: 0.720	 OF1: 0.780	CP: 0.834	 CR: 0.669	 CF1: 0.743
OP_3: 0.893	 OR_3: 0.640	 OF1_3: 0.746	CP_3: 0.867	 CR_3: 0.596	 CF1_3: 0.706
Save checkpoint to /workspace/cluster/HDD/azuma/Others/github/ML_DL_Notebook/MultiLabel_Image_Recognition/ADD-GCN/results/checkpoint.pth
Save results to /workspace/cluster/HDD/azuma/Others/github/ML_DL_Notebook/MultiLabel_Image_Recognition/ADD-GCN/results/results.csv
 * best mAP=0.8034
Lr: [0.005 0.05 ]
2025-09-23 05:03:43, 6 Epoch, 0 Iter, Loss: 0.0512, Data time: 0.9393, Batch time: 1.1338
2025-09-23 05:04:58, 6 Epoch, 200 Iter, Loss: 0.0285, Data time: 0.0902, Batch time: 0.1998
2025-09-23 05:06:13, 6 Epoch, 400 Iter, Loss: 0.0426, Data time: 0.0806, Batch time: 0.1973
2025-09-23 05:07:28, 6 Epoch, 600 Iter, Loss: 0.0432, Data time: 0.0229, Batch time: 0.1277
2025-09-23 05:08:44, 6 Epoch, 800 Iter, Loss: 0.0456, Data time: 0.0856, Batch time: 0.2023
2025-09-23 05:10:00, 6 Epoch, 1000 Iter, Loss: 0.0327, Data time: 0.1325, Batch time: 0.3221
2025-09-23 05:11:16, 6 Epoch, 1200 Iter, Loss: 0.0589, Data time: 0.1205, Batch time: 0.4153
2025-09-23 05:12:33, 6 Epoch, 1400 Iter, Loss: 0.0277, Data time: 0.0394, Batch time: 0.2304
2025-09-23 05:13:48, 6 Epoch, 1600 Iter, Loss: 0.0418, Data time: 0.1097, Batch time: 0.2337
2025-09-23 05:15:04, 6 Epoch, 1800 Iter, Loss: 0.0523, Data time: 0.1178, Batch time: 0.3232
2025-09-23 05:16:19, 6 Epoch, 2000 Iter, Loss: 0.0419, Data time: 0.0323, Batch time: 0.1387
2025-09-23 05:17:35, 6 Epoch, 2200 Iter, Loss: 0.0461, Data time: 0.0191, Batch time: 0.2210
2025-09-23 05:18:51, 6 Epoch, 2400 Iter, Loss: 0.0470, Data time: 0.1075, Batch time: 0.2317
2025-09-23 05:20:07, 6 Epoch, 2600 Iter, Loss: 0.0509, Data time: 0.0302, Batch time: 0.1386
2025-09-23 05:21:23, 6 Epoch, 2800 Iter, Loss: 0.0289, Data time: 0.0430, Batch time: 0.3091
2025-09-23 05:22:39, 6 Epoch, 3000 Iter, Loss: 0.0367, Data time: 0.0427, Batch time: 0.2278
2025-09-23 05:23:54, 6 Epoch, 3200 Iter, Loss: 0.0645, Data time: 0.0954, Batch time: 0.2180
2025-09-23 05:25:10, 6 Epoch, 3400 Iter, Loss: 0.0393, Data time: 0.0201, Batch time: 0.1294
2025-09-23 05:26:27, 6 Epoch, 3600 Iter, Loss: 0.0577, Data time: 0.0111, Batch time: 0.1284
2025-09-23 05:27:45, 6 Epoch, 3800 Iter, Loss: 0.0565, Data time: 0.1836, Batch time: 0.4775
2025-09-23 05:29:00, 6 Epoch, 4000 Iter, Loss: 0.0392, Data time: 0.0453, Batch time: 0.2303
2025-09-23 05:30:17, 6 Epoch, 4200 Iter, Loss: 0.0375, Data time: 0.1115, Batch time: 0.2338
2025-09-23 05:31:33, 6 Epoch, 4400 Iter, Loss: 0.0361, Data time: 0.0403, Batch time: 0.2283
2025-09-23 05:32:49, 6 Epoch, 4600 Iter, Loss: 0.0404, Data time: 0.3023, Batch time: 0.5130
2025-09-23 05:34:05, 6 Epoch, 4800 Iter, Loss: 0.0492, Data time: 0.0229, Batch time: 0.1277
2025-09-23 05:35:21, 6 Epoch, 5000 Iter, Loss: 0.0380, Data time: 0.0284, Batch time: 0.1348
Validate: 100%|█████████████████████████████| 2509/2509 [11:03<00:00,  3.78it/s]
tensor([0.9653, 0.6764, 0.4856, 0.8417, 0.9407, 0.9432, 0.9806, 0.8575, 0.6398,
        0.7768, 0.8128, 0.8881, 0.6681, 0.7300, 0.7312, 0.9359, 0.8798, 0.8128,
        0.8653, 0.7681, 0.9597, 0.6607, 0.7786, 0.8070, 0.7962, 0.9167, 0.7617,
        0.7902, 0.8977, 0.8394, 0.9825, 0.8376, 0.7712, 0.9284, 0.9951, 0.1441,
        0.5298, 0.9253, 0.7536, 0.8753, 0.9658, 0.6280, 0.8814, 0.7708, 0.9304,
        0.8713, 0.8132, 0.8501, 0.6737, 0.9904, 0.9369, 0.6788, 0.7445, 0.7650,
        0.7349, 0.6299, 0.9455, 0.8665, 0.9680, 0.9341, 0.8371, 0.5983, 0.8591,
        0.7721, 0.7025, 0.9464, 0.8677, 0.9887, 0.8395, 0.0969, 0.9763, 0.6774,
        0.8421, 0.9618, 0.7157, 0.8660, 0.8471, 0.7806, 0.7587, 0.9931])
* Test
Loss: 0.0427	 mAP: 0.8082	Data_time: 0.1514	 Batch_time: 0.2641
OP: 0.883	 OR: 0.699	 OF1: 0.780	CP: 0.854	 CR: 0.658	 CF1: 0.744
OP_3: 0.916	 OR_3: 0.631	 OF1_3: 0.747	CP_3: 0.880	 CR_3: 0.597	 CF1_3: 0.712
Save checkpoint to /workspace/cluster/HDD/azuma/Others/github/ML_DL_Notebook/MultiLabel_Image_Recognition/ADD-GCN/results/checkpoint.pth
Save results to /workspace/cluster/HDD/azuma/Others/github/ML_DL_Notebook/MultiLabel_Image_Recognition/ADD-GCN/results/results.csv
 * best mAP=0.8082
Lr: [0.005 0.05 ]
2025-09-23 05:48:56, 7 Epoch, 0 Iter, Loss: 0.0563, Data time: 1.0277, Batch time: 1.1658
2025-09-23 05:50:11, 7 Epoch, 200 Iter, Loss: 0.0343, Data time: 0.0193, Batch time: 0.1298
2025-09-23 05:51:26, 7 Epoch, 400 Iter, Loss: 0.0281, Data time: 0.1017, Batch time: 0.2300
2025-09-23 05:52:41, 7 Epoch, 600 Iter, Loss: 0.0505, Data time: 0.1259, Batch time: 0.3117
2025-09-23 05:53:56, 7 Epoch, 800 Iter, Loss: 0.0428, Data time: 0.0279, Batch time: 0.1338
2025-09-23 05:55:13, 7 Epoch, 1000 Iter, Loss: 0.0697, Data time: 0.0320, Batch time: 0.2195
2025-09-23 05:56:28, 7 Epoch, 1200 Iter, Loss: 0.0380, Data time: 0.1016, Batch time: 0.3032
2025-09-23 05:57:44, 7 Epoch, 1400 Iter, Loss: 0.0535, Data time: 0.0205, Batch time: 0.1295
2025-09-23 05:58:59, 7 Epoch, 1600 Iter, Loss: 0.0192, Data time: 0.0337, Batch time: 0.1482
2025-09-23 06:00:15, 7 Epoch, 1800 Iter, Loss: 0.0479, Data time: 0.1245, Batch time: 0.3216
2025-09-23 06:01:31, 7 Epoch, 2000 Iter, Loss: 0.0524, Data time: 0.0927, Batch time: 0.1988
2025-09-23 06:02:46, 7 Epoch, 2200 Iter, Loss: 0.0360, Data time: 0.1249, Batch time: 0.4041
2025-09-23 06:04:01, 7 Epoch, 2400 Iter, Loss: 0.0453, Data time: 0.0353, Batch time: 0.2168
2025-09-23 06:05:17, 7 Epoch, 2600 Iter, Loss: 0.0309, Data time: 0.0421, Batch time: 0.2316
2025-09-23 06:06:32, 7 Epoch, 2800 Iter, Loss: 0.0566, Data time: 0.0362, Batch time: 0.2167
2025-09-23 06:07:48, 7 Epoch, 3000 Iter, Loss: 0.0620, Data time: 0.0343, Batch time: 0.2213
2025-09-23 06:09:03, 7 Epoch, 3200 Iter, Loss: 0.0768, Data time: 0.1010, Batch time: 0.2906
2025-09-23 06:10:21, 7 Epoch, 3400 Iter, Loss: 0.0276, Data time: 0.0261, Batch time: 0.3038
2025-09-23 06:11:38, 7 Epoch, 3600 Iter, Loss: 0.0402, Data time: 0.1011, Batch time: 0.2299
2025-09-23 06:12:54, 7 Epoch, 3800 Iter, Loss: 0.0570, Data time: 0.0412, Batch time: 0.2288
2025-09-23 06:14:09, 7 Epoch, 4000 Iter, Loss: 0.0421, Data time: 0.0812, Batch time: 0.1997
2025-09-23 06:15:26, 7 Epoch, 4200 Iter, Loss: 0.0353, Data time: 0.0937, Batch time: 0.2070
2025-09-23 06:16:41, 7 Epoch, 4400 Iter, Loss: 0.0429, Data time: 0.0174, Batch time: 0.1297
2025-09-23 06:17:56, 7 Epoch, 4600 Iter, Loss: 0.0433, Data time: 0.0986, Batch time: 0.2049
2025-09-23 06:19:13, 7 Epoch, 4800 Iter, Loss: 0.0379, Data time: 0.0367, Batch time: 0.2225
2025-09-23 06:20:29, 7 Epoch, 5000 Iter, Loss: 0.0409, Data time: 0.0426, Batch time: 0.2294
Validate: 100%|█████████████████████████████| 2509/2509 [11:12<00:00,  3.73it/s]
tensor([0.9607, 0.6710, 0.5003, 0.8360, 0.9427, 0.9522, 0.9800, 0.8573, 0.6605,
        0.7878, 0.8035, 0.8735, 0.6796, 0.7303, 0.7324, 0.9287, 0.8809, 0.8197,
        0.8698, 0.7793, 0.9612, 0.6718, 0.7783, 0.8091, 0.7803, 0.9232, 0.7677,
        0.7897, 0.8976, 0.8310, 0.9801, 0.8466, 0.7804, 0.9338, 0.9947, 0.3673,
        0.5424, 0.9353, 0.7724, 0.8827, 0.9711, 0.6549, 0.8832, 0.7803, 0.9295,
        0.8664, 0.8084, 0.8582, 0.6689, 0.9904, 0.9446, 0.6846, 0.7483, 0.7670,
        0.7603, 0.6177, 0.9455, 0.8869, 0.9668, 0.9276, 0.8440, 0.6267, 0.8533,
        0.7770, 0.7063, 0.9414, 0.8736, 0.9908, 0.8455, 0.1057, 0.9768, 0.6980,
        0.8348, 0.9598, 0.7321, 0.8571, 0.8588, 0.7903, 0.7749, 0.9933])
* Test
Loss: 0.0422	 mAP: 0.8149	Data_time: 0.1547	 Batch_time: 0.2676
OP: 0.887	 OR: 0.699	 OF1: 0.781	CP: 0.877	 CR: 0.655	 CF1: 0.750
OP_3: 0.919	 OR_3: 0.631	 OF1_3: 0.748	CP_3: 0.900	 CR_3: 0.595	 CF1_3: 0.716
Save checkpoint to /workspace/cluster/HDD/azuma/Others/github/ML_DL_Notebook/MultiLabel_Image_Recognition/ADD-GCN/results/checkpoint.pth
Save results to /workspace/cluster/HDD/azuma/Others/github/ML_DL_Notebook/MultiLabel_Image_Recognition/ADD-GCN/results/results.csv
 * best mAP=0.8149
Lr: [0.005 0.05 ]
2025-09-23 06:34:12, 8 Epoch, 0 Iter, Loss: 0.0461, Data time: 1.0065, Batch time: 1.2095
2025-09-23 06:35:34, 8 Epoch, 200 Iter, Loss: 0.0348, Data time: 0.0465, Batch time: 0.2282
2025-09-23 06:36:57, 8 Epoch, 400 Iter, Loss: 0.0291, Data time: 0.1077, Batch time: 0.2318
2025-09-23 06:38:20, 8 Epoch, 600 Iter, Loss: 0.0634, Data time: 0.1062, Batch time: 0.3040
2025-09-23 06:39:43, 8 Epoch, 800 Iter, Loss: 0.0523, Data time: 0.1042, Batch time: 0.2321
2025-09-23 06:41:07, 8 Epoch, 1000 Iter, Loss: 0.0695, Data time: 0.1182, Batch time: 0.2992
2025-09-23 06:42:30, 8 Epoch, 1200 Iter, Loss: 0.0355, Data time: 0.1847, Batch time: 0.4739
2025-09-23 06:43:53, 8 Epoch, 1400 Iter, Loss: 0.0417, Data time: 0.1045, Batch time: 0.2139
2025-09-23 06:45:17, 8 Epoch, 1600 Iter, Loss: 0.0410, Data time: 0.2796, Batch time: 0.5750
2025-09-23 06:46:40, 8 Epoch, 1800 Iter, Loss: 0.0547, Data time: 0.1744, Batch time: 0.3004
2025-09-23 06:48:03, 8 Epoch, 2000 Iter, Loss: 0.0778, Data time: 0.1066, Batch time: 0.2723
2025-09-23 06:49:26, 8 Epoch, 2200 Iter, Loss: 0.0418, Data time: 0.0313, Batch time: 0.1379
2025-09-23 06:50:50, 8 Epoch, 2400 Iter, Loss: 0.0480, Data time: 0.1072, Batch time: 0.2316
2025-09-23 06:52:13, 8 Epoch, 2600 Iter, Loss: 0.0611, Data time: 0.1082, Batch time: 0.2971
2025-09-23 06:53:37, 8 Epoch, 2800 Iter, Loss: 0.0568, Data time: 0.1698, Batch time: 0.2991
2025-09-23 06:55:01, 8 Epoch, 3000 Iter, Loss: 0.0333, Data time: 0.1084, Batch time: 0.2270
2025-09-23 06:56:24, 8 Epoch, 3200 Iter, Loss: 0.0515, Data time: 0.1062, Batch time: 0.2747
2025-09-23 06:57:48, 8 Epoch, 3400 Iter, Loss: 0.0393, Data time: 0.1906, Batch time: 0.4728
2025-09-23 06:59:10, 8 Epoch, 3600 Iter, Loss: 0.0635, Data time: 0.2039, Batch time: 0.3287
2025-09-23 07:00:33, 8 Epoch, 3800 Iter, Loss: 0.0571, Data time: 0.2149, Batch time: 0.4147
2025-09-23 07:01:56, 8 Epoch, 4000 Iter, Loss: 0.0298, Data time: 0.0375, Batch time: 0.2261
2025-09-23 07:03:18, 8 Epoch, 4200 Iter, Loss: 0.0419, Data time: 0.2009, Batch time: 0.5018
2025-09-23 07:04:41, 8 Epoch, 4400 Iter, Loss: 0.0659, Data time: 0.1328, Batch time: 0.3221
2025-09-23 07:06:04, 8 Epoch, 4600 Iter, Loss: 0.0691, Data time: 0.0986, Batch time: 0.2037
2025-09-23 07:07:28, 8 Epoch, 4800 Iter, Loss: 0.0433, Data time: 0.1070, Batch time: 0.2292
2025-09-23 07:08:51, 8 Epoch, 5000 Iter, Loss: 0.0475, Data time: 0.1126, Batch time: 0.2335
Validate: 100%|█████████████████████████████| 2509/2509 [12:22<00:00,  3.38it/s]
tensor([0.9640, 0.6942, 0.4964, 0.8369, 0.9474, 0.9520, 0.9787, 0.8680, 0.6656,
        0.7970, 0.8109, 0.8804, 0.6792, 0.7289, 0.7410, 0.9345, 0.8786, 0.8216,
        0.8808, 0.7981, 0.9637, 0.6602, 0.7814, 0.8136, 0.8097, 0.9200, 0.7570,
        0.7879, 0.8989, 0.8470, 0.9801, 0.8395, 0.7755, 0.9285, 0.9962, 0.3540,
        0.5473, 0.9332, 0.7821, 0.8795, 0.9618, 0.6052, 0.8856, 0.7807, 0.9283,
        0.8621, 0.8074, 0.8606, 0.6930, 0.9906, 0.9415, 0.6891, 0.7457, 0.7772,
        0.7590, 0.6144, 0.9391, 0.8971, 0.9651, 0.9259, 0.8435, 0.6327, 0.8626,
        0.7713, 0.7092, 0.9422, 0.8671, 0.9904, 0.8434, 0.1012, 0.9761, 0.7098,
        0.8513, 0.9635, 0.7331, 0.8575, 0.8574, 0.7817, 0.7764, 0.9920])
* Test
Loss: 0.0426	 mAP: 0.8163	Data_time: 0.1807	 Batch_time: 0.2955
OP: 0.862	 OR: 0.723	 OF1: 0.786	CP: 0.851	 CR: 0.682	 CF1: 0.757
OP_3: 0.899	 OR_3: 0.645	 OF1_3: 0.751	CP_3: 0.879	 CR_3: 0.612	 CF1_3: 0.722
Save checkpoint to /workspace/cluster/HDD/azuma/Others/github/ML_DL_Notebook/MultiLabel_Image_Recognition/ADD-GCN/results/checkpoint.pth
Save results to /workspace/cluster/HDD/azuma/Others/github/ML_DL_Notebook/MultiLabel_Image_Recognition/ADD-GCN/results/results.csv
 * best mAP=0.8163
Lr: [0.005 0.05 ]
2025-09-23 07:23:48, 9 Epoch, 0 Iter, Loss: 0.0367, Data time: 1.0176, Batch time: 1.2826
2025-09-23 07:25:11, 9 Epoch, 200 Iter, Loss: 0.0495, Data time: 0.2021, Batch time: 0.3308
2025-09-23 07:26:33, 9 Epoch, 400 Iter, Loss: 0.0504, Data time: 0.1175, Batch time: 0.2356
2025-09-23 07:27:56, 9 Epoch, 600 Iter, Loss: 0.0397, Data time: 0.0412, Batch time: 0.2295
2025-09-23 07:29:19, 9 Epoch, 800 Iter, Loss: 0.0203, Data time: 0.1165, Batch time: 0.2390
2025-09-23 07:30:42, 9 Epoch, 1000 Iter, Loss: 0.0454, Data time: 0.1112, Batch time: 0.3723
2025-09-23 07:32:05, 9 Epoch, 1200 Iter, Loss: 0.0258, Data time: 0.1177, Batch time: 0.4056
2025-09-23 07:33:28, 9 Epoch, 1400 Iter, Loss: 0.0639, Data time: 0.2013, Batch time: 0.4263
2025-09-23 07:34:51, 9 Epoch, 1600 Iter, Loss: 0.0395, Data time: 0.1027, Batch time: 0.2298
2025-09-23 07:36:14, 9 Epoch, 1800 Iter, Loss: 0.0474, Data time: 0.1126, Batch time: 0.2320
2025-09-23 07:37:37, 9 Epoch, 2000 Iter, Loss: 0.0502, Data time: 0.1330, Batch time: 0.3227
2025-09-23 07:39:00, 9 Epoch, 2200 Iter, Loss: 0.0612, Data time: 0.0440, Batch time: 0.2273
2025-09-23 07:40:23, 9 Epoch, 2400 Iter, Loss: 0.0586, Data time: 0.1821, Batch time: 0.2985
2025-09-23 07:41:46, 9 Epoch, 2600 Iter, Loss: 0.0327, Data time: 0.1787, Batch time: 0.3026
2025-09-23 07:43:09, 9 Epoch, 2800 Iter, Loss: 0.0666, Data time: 0.0260, Batch time: 0.1311
2025-09-23 07:44:31, 9 Epoch, 3000 Iter, Loss: 0.0589, Data time: 0.1085, Batch time: 0.3007
2025-09-23 07:45:54, 9 Epoch, 3200 Iter, Loss: 0.0514, Data time: 0.2074, Batch time: 0.4188
2025-09-23 07:47:17, 9 Epoch, 3400 Iter, Loss: 0.0395, Data time: 0.1697, Batch time: 0.2996
2025-09-23 07:48:41, 9 Epoch, 3600 Iter, Loss: 0.0724, Data time: 0.1085, Batch time: 0.3012
2025-09-23 07:50:06, 9 Epoch, 3800 Iter, Loss: 0.0496, Data time: 0.0472, Batch time: 0.2298
2025-09-23 07:51:28, 9 Epoch, 4000 Iter, Loss: 0.0493, Data time: 0.1101, Batch time: 0.2329
2025-09-23 07:52:51, 9 Epoch, 4200 Iter, Loss: 0.0556, Data time: 0.1056, Batch time: 0.2725
2025-09-23 07:54:15, 9 Epoch, 4400 Iter, Loss: 0.0417, Data time: 0.1068, Batch time: 0.3056
2025-09-23 07:55:37, 9 Epoch, 4600 Iter, Loss: 0.0444, Data time: 0.2017, Batch time: 0.3374
2025-09-23 07:57:00, 9 Epoch, 4800 Iter, Loss: 0.0362, Data time: 0.1725, Batch time: 0.3692
2025-09-23 07:58:23, 9 Epoch, 5000 Iter, Loss: 0.0734, Data time: 0.0423, Batch time: 0.2285
Validate: 100%|█████████████████████████████| 2509/2509 [12:24<00:00,  3.37it/s]
tensor([0.9591, 0.6898, 0.5020, 0.8396, 0.9458, 0.9469, 0.9750, 0.8612, 0.6601,
        0.7893, 0.8093, 0.8837, 0.6855, 0.7366, 0.7412, 0.9298, 0.8627, 0.8364,
        0.8802, 0.8038, 0.9596, 0.6770, 0.7823, 0.8083, 0.8071, 0.9167, 0.7644,
        0.7828, 0.9046, 0.8549, 0.9800, 0.8511, 0.7830, 0.9300, 0.9933, 0.4039,
        0.5447, 0.9305, 0.7907, 0.8829, 0.9707, 0.6723, 0.8867, 0.7888, 0.9343,
        0.8745, 0.8107, 0.8631, 0.6977, 0.9905, 0.9449, 0.6864, 0.7695, 0.7623,
        0.7561, 0.6456, 0.9504, 0.8930, 0.9716, 0.9402, 0.8622, 0.6270, 0.8574,
        0.7750, 0.7134, 0.9529, 0.8692, 0.9904, 0.8483, 0.1388, 0.9746, 0.7222,
        0.8480, 0.9614, 0.7254, 0.8705, 0.8570, 0.7903, 0.7725, 0.9931])
* Test
Loss: 0.0418	 mAP: 0.8206	Data_time: 0.1822	 Batch_time: 0.2966
OP: 0.867	 OR: 0.722	 OF1: 0.788	CP: 0.850	 CR: 0.688	 CF1: 0.761
OP_3: 0.907	 OR_3: 0.644	 OF1_3: 0.753	CP_3: 0.881	 CR_3: 0.618	 CF1_3: 0.727
Save checkpoint to /workspace/cluster/HDD/azuma/Others/github/ML_DL_Notebook/MultiLabel_Image_Recognition/ADD-GCN/results/checkpoint.pth
Save results to /workspace/cluster/HDD/azuma/Others/github/ML_DL_Notebook/MultiLabel_Image_Recognition/ADD-GCN/results/results.csv
 * best mAP=0.8206
Lr: [0.005 0.05 ]
2025-09-23 08:13:22, 10 Epoch, 0 Iter, Loss: 0.0435, Data time: 0.9089, Batch time: 1.1039
2025-09-23 08:14:37, 10 Epoch, 200 Iter, Loss: 0.0376, Data time: 0.0868, Batch time: 0.2017
2025-09-23 08:15:53, 10 Epoch, 400 Iter, Loss: 0.0549, Data time: 0.1118, Batch time: 0.3010
2025-09-23 08:17:08, 10 Epoch, 600 Iter, Loss: 0.0562, Data time: 0.1197, Batch time: 0.2348
2025-09-23 08:18:23, 10 Epoch, 800 Iter, Loss: 0.0669, Data time: 0.1028, Batch time: 0.2316
2025-09-23 08:19:40, 10 Epoch, 1000 Iter, Loss: 0.0517, Data time: 0.0419, Batch time: 0.3030
2025-09-23 08:20:56, 10 Epoch, 1200 Iter, Loss: 0.0535, Data time: 0.0969, Batch time: 0.2025
2025-09-23 08:22:12, 10 Epoch, 1400 Iter, Loss: 0.0408, Data time: 0.0418, Batch time: 0.2300
2025-09-23 08:23:27, 10 Epoch, 1600 Iter, Loss: 0.0551, Data time: 0.1169, Batch time: 0.3185
2025-09-23 08:24:43, 10 Epoch, 1800 Iter, Loss: 0.0508, Data time: 0.0934, Batch time: 0.1973
2025-09-23 08:26:00, 10 Epoch, 2000 Iter, Loss: 0.0434, Data time: 0.1292, Batch time: 0.3322
2025-09-23 08:27:15, 10 Epoch, 2200 Iter, Loss: 0.0315, Data time: 0.0870, Batch time: 0.2032
2025-09-23 08:28:30, 10 Epoch, 2400 Iter, Loss: 0.0340, Data time: 0.0309, Batch time: 0.1349
2025-09-23 08:29:45, 10 Epoch, 2600 Iter, Loss: 0.0223, Data time: 0.0206, Batch time: 0.1296
2025-09-23 08:31:01, 10 Epoch, 2800 Iter, Loss: 0.0399, Data time: 0.1006, Batch time: 0.2318
2025-09-23 08:32:17, 10 Epoch, 3000 Iter, Loss: 0.0384, Data time: 0.1088, Batch time: 0.2975
2025-09-23 08:33:34, 10 Epoch, 3200 Iter, Loss: 0.0329, Data time: 0.0847, Batch time: 0.1998
2025-09-23 08:34:50, 10 Epoch, 3400 Iter, Loss: 0.0481, Data time: 0.0303, Batch time: 0.2033
2025-09-23 08:36:06, 10 Epoch, 3600 Iter, Loss: 0.0361, Data time: 0.1007, Batch time: 0.2300
2025-09-23 08:37:21, 10 Epoch, 3800 Iter, Loss: 0.0505, Data time: 0.1115, Batch time: 0.3249
2025-09-23 08:38:35, 10 Epoch, 4000 Iter, Loss: 0.0541, Data time: 0.0174, Batch time: 0.2225
2025-09-23 08:39:50, 10 Epoch, 4200 Iter, Loss: 0.0355, Data time: 0.1280, Batch time: 0.2379
2025-09-23 08:41:06, 10 Epoch, 4400 Iter, Loss: 0.0370, Data time: 0.1155, Batch time: 0.4052
2025-09-23 08:42:21, 10 Epoch, 4600 Iter, Loss: 0.0501, Data time: 0.1010, Batch time: 0.2293
2025-09-23 08:43:36, 10 Epoch, 4800 Iter, Loss: 0.0320, Data time: 0.0249, Batch time: 0.1189
2025-09-23 08:44:51, 10 Epoch, 5000 Iter, Loss: 0.0461, Data time: 0.1022, Batch time: 0.2300
Validate: 100%|█████████████████████████████| 2509/2509 [11:00<00:00,  3.80it/s]
tensor([0.9625, 0.6935, 0.5080, 0.8495, 0.9499, 0.9484, 0.9779, 0.8688, 0.6619,
        0.7851, 0.8092, 0.8824, 0.6911, 0.7423, 0.7325, 0.9354, 0.8783, 0.8297,
        0.8742, 0.7863, 0.9624, 0.6699, 0.7735, 0.8135, 0.8119, 0.9175, 0.7537,
        0.7875, 0.8960, 0.8483, 0.9808, 0.8467, 0.7808, 0.9268, 0.9948, 0.3622,
        0.5426, 0.9290, 0.7977, 0.8888, 0.9681, 0.6668, 0.8846, 0.7911, 0.9345,
        0.8655, 0.8113, 0.8617, 0.6899, 0.9903, 0.9423, 0.6911, 0.7635, 0.7815,
        0.7494, 0.6106, 0.9424, 0.8944, 0.9707, 0.9434, 0.8538, 0.6381, 0.8545,
        0.7823, 0.7100, 0.9521, 0.8650, 0.9897, 0.8470, 0.1958, 0.9703, 0.7258,
        0.8474, 0.9597, 0.7207, 0.8722, 0.8609, 0.7909, 0.7748, 0.9925])
* Test
Loss: 0.0429	 mAP: 0.8201	Data_time: 0.1509	 Batch_time: 0.2630
OP: 0.857	 OR: 0.725	 OF1: 0.785	CP: 0.839	 CR: 0.696	 CF1: 0.761
OP_3: 0.899	 OR_3: 0.644	 OF1_3: 0.751	CP_3: 0.872	 CR_3: 0.626	 CF1_3: 0.729
Save checkpoint to /workspace/cluster/HDD/azuma/Others/github/ML_DL_Notebook/MultiLabel_Image_Recognition/ADD-GCN/results/checkpoint.pth
Save results to /workspace/cluster/HDD/azuma/Others/github/ML_DL_Notebook/MultiLabel_Image_Recognition/ADD-GCN/results/results.csv
 * best mAP=0.8206
Lr: [0.005 0.05 ]
2025-09-23 08:58:19, 11 Epoch, 0 Iter, Loss: 0.0759, Data time: 0.7882, Batch time: 1.0548
2025-09-23 08:59:34, 11 Epoch, 200 Iter, Loss: 0.0686, Data time: 0.1270, Batch time: 0.4115
2025-09-23 09:00:49, 11 Epoch, 400 Iter, Loss: 0.0563, Data time: 0.0206, Batch time: 0.1295
2025-09-23 09:02:05, 11 Epoch, 600 Iter, Loss: 0.0389, Data time: 0.1297, Batch time: 0.2347
2025-09-23 09:03:21, 11 Epoch, 800 Iter, Loss: 0.0460, Data time: 0.0406, Batch time: 0.2276
2025-09-23 09:04:37, 11 Epoch, 1000 Iter, Loss: 0.0602, Data time: 0.0872, Batch time: 0.2007
2025-09-23 09:05:53, 11 Epoch, 1200 Iter, Loss: 0.0268, Data time: 0.0250, Batch time: 0.1301
2025-09-23 09:07:08, 11 Epoch, 1400 Iter, Loss: 0.0376, Data time: 0.1057, Batch time: 0.2751
2025-09-23 09:08:23, 11 Epoch, 1600 Iter, Loss: 0.0523, Data time: 0.0779, Batch time: 0.2024
2025-09-23 09:09:39, 11 Epoch, 1800 Iter, Loss: 0.0483, Data time: 0.0182, Batch time: 0.3053
2025-09-23 09:10:55, 11 Epoch, 2000 Iter, Loss: 0.0378, Data time: 0.1238, Batch time: 0.3317
2025-09-23 09:12:10, 11 Epoch, 2200 Iter, Loss: 0.0428, Data time: 0.0332, Batch time: 0.2055
2025-09-23 09:13:26, 11 Epoch, 2400 Iter, Loss: 0.0537, Data time: 0.0369, Batch time: 0.2170
2025-09-23 09:14:42, 11 Epoch, 2600 Iter, Loss: 0.0537, Data time: 0.0362, Batch time: 0.2151
2025-09-23 09:15:58, 11 Epoch, 2800 Iter, Loss: 0.0569, Data time: 0.1688, Batch time: 0.2982
2025-09-23 09:17:15, 11 Epoch, 3000 Iter, Loss: 0.0654, Data time: 0.3014, Batch time: 0.4337
2025-09-23 09:18:31, 11 Epoch, 3200 Iter, Loss: 0.0368, Data time: 0.0238, Batch time: 0.1295
2025-09-23 09:19:47, 11 Epoch, 3400 Iter, Loss: 0.0269, Data time: 0.0469, Batch time: 0.2293
2025-09-23 09:21:03, 11 Epoch, 3600 Iter, Loss: 0.0574, Data time: 0.0402, Batch time: 0.2275
2025-09-23 09:22:19, 11 Epoch, 3800 Iter, Loss: 0.0391, Data time: 0.1146, Batch time: 0.2973
2025-09-23 09:23:35, 11 Epoch, 4000 Iter, Loss: 0.0623, Data time: 0.1199, Batch time: 0.3307
2025-09-23 09:24:51, 11 Epoch, 4200 Iter, Loss: 0.0508, Data time: 0.0867, Batch time: 0.2038
2025-09-23 09:26:07, 11 Epoch, 4400 Iter, Loss: 0.0472, Data time: 0.0966, Batch time: 0.2011
2025-09-23 09:27:23, 11 Epoch, 4600 Iter, Loss: 0.0391, Data time: 0.0438, Batch time: 0.2306
2025-09-23 09:28:38, 11 Epoch, 4800 Iter, Loss: 0.0318, Data time: 0.0990, Batch time: 0.2045
2025-09-23 09:29:54, 11 Epoch, 5000 Iter, Loss: 0.0495, Data time: 0.0399, Batch time: 0.2294
Validate: 100%|█████████████████████████████| 2509/2509 [11:00<00:00,  3.80it/s]
tensor([0.9620, 0.6873, 0.5138, 0.8515, 0.9466, 0.9482, 0.9666, 0.8656, 0.6703,
        0.7897, 0.8136, 0.8843, 0.6795, 0.7399, 0.7315, 0.9359, 0.8764, 0.8212,
        0.8821, 0.7954, 0.9614, 0.6773, 0.7859, 0.8090, 0.8072, 0.9216, 0.7688,
        0.7836, 0.8908, 0.8473, 0.9824, 0.8440, 0.7775, 0.9368, 0.9957, 0.3567,
        0.5565, 0.9332, 0.7781, 0.8836, 0.9778, 0.6575, 0.8866, 0.7796, 0.9346,
        0.8665, 0.8053, 0.8705, 0.6771, 0.9907, 0.9373, 0.6961, 0.7785, 0.7906,
        0.7612, 0.6315, 0.9545, 0.8945, 0.9667, 0.9422, 0.8341, 0.6306, 0.8648,
        0.7862, 0.7154, 0.9496, 0.8769, 0.9923, 0.8400, 0.3029, 0.9774, 0.7181,
        0.8492, 0.9590, 0.7286, 0.8702, 0.8551, 0.7936, 0.7727, 0.9931])
* Test
Loss: 0.0420	 mAP: 0.8221	Data_time: 0.1505	 Batch_time: 0.2632
OP: 0.885	 OR: 0.714	 OF1: 0.790	CP: 0.875	 CR: 0.680	 CF1: 0.765
OP_3: 0.916	 OR_3: 0.642	 OF1_3: 0.755	CP_3: 0.903	 CR_3: 0.615	 CF1_3: 0.731
Save checkpoint to /workspace/cluster/HDD/azuma/Others/github/ML_DL_Notebook/MultiLabel_Image_Recognition/ADD-GCN/results/checkpoint.pth
Save results to /workspace/cluster/HDD/azuma/Others/github/ML_DL_Notebook/MultiLabel_Image_Recognition/ADD-GCN/results/results.csv
 * best mAP=0.8221
Lr: [0.005 0.05 ]
2025-09-23 09:43:27, 12 Epoch, 0 Iter, Loss: 0.0500, Data time: 1.1398, Batch time: 1.3387
2025-09-23 09:44:41, 12 Epoch, 200 Iter, Loss: 0.0594, Data time: 0.2058, Batch time: 0.4086
2025-09-23 09:45:58, 12 Epoch, 400 Iter, Loss: 0.0589, Data time: 0.0292, Batch time: 0.2282
2025-09-23 09:47:13, 12 Epoch, 600 Iter, Loss: 0.0998, Data time: 0.0455, Batch time: 0.3048
2025-09-23 09:48:27, 12 Epoch, 800 Iter, Loss: 0.0506, Data time: 0.0835, Batch time: 0.2006
2025-09-23 09:49:43, 12 Epoch, 1000 Iter, Loss: 0.0282, Data time: 0.1225, Batch time: 0.3263
2025-09-23 09:50:58, 12 Epoch, 1200 Iter, Loss: 0.0517, Data time: 0.1113, Batch time: 0.3187
2025-09-23 09:52:13, 12 Epoch, 1400 Iter, Loss: 0.0386, Data time: 0.1183, Batch time: 0.3222
2025-09-23 09:53:29, 12 Epoch, 1600 Iter, Loss: 0.0535, Data time: 0.1013, Batch time: 0.2267
2025-09-23 09:54:43, 12 Epoch, 1800 Iter, Loss: 0.0405, Data time: 0.1217, Batch time: 0.3321
2025-09-23 09:55:58, 12 Epoch, 2000 Iter, Loss: 0.0514, Data time: 0.0462, Batch time: 0.2297
2025-09-23 09:57:13, 12 Epoch, 2200 Iter, Loss: 0.0353, Data time: 0.0438, Batch time: 0.2294
2025-09-23 09:58:28, 12 Epoch, 2400 Iter, Loss: 0.0601, Data time: 0.0364, Batch time: 0.2296
2025-09-23 09:59:44, 12 Epoch, 2600 Iter, Loss: 0.0432, Data time: 0.1840, Batch time: 0.2932
2025-09-23 10:01:00, 12 Epoch, 2800 Iter, Loss: 0.0383, Data time: 0.1068, Batch time: 0.2335
2025-09-23 10:02:15, 12 Epoch, 3000 Iter, Loss: 0.0519, Data time: 0.1177, Batch time: 0.3153
2025-09-23 10:03:29, 12 Epoch, 3200 Iter, Loss: 0.0369, Data time: 0.0326, Batch time: 0.2101
2025-09-23 10:04:45, 12 Epoch, 3400 Iter, Loss: 0.0469, Data time: 0.0179, Batch time: 0.3044
2025-09-23 10:06:00, 12 Epoch, 3600 Iter, Loss: 0.0534, Data time: 0.0238, Batch time: 0.2289
2025-09-23 10:07:16, 12 Epoch, 3800 Iter, Loss: 0.0303, Data time: 0.1300, Batch time: 0.4027
2025-09-23 10:08:32, 12 Epoch, 4000 Iter, Loss: 0.0355, Data time: 0.1059, Batch time: 0.2341
2025-09-23 10:09:47, 12 Epoch, 4200 Iter, Loss: 0.0342, Data time: 0.0236, Batch time: 0.1346
2025-09-23 10:11:03, 12 Epoch, 4400 Iter, Loss: 0.0472, Data time: 0.0170, Batch time: 0.1281
2025-09-23 10:12:18, 12 Epoch, 4600 Iter, Loss: 0.0226, Data time: 0.0218, Batch time: 0.1279
2025-09-23 10:13:34, 12 Epoch, 4800 Iter, Loss: 0.0676, Data time: 0.1120, Batch time: 0.2973
2025-09-23 10:14:50, 12 Epoch, 5000 Iter, Loss: 0.0255, Data time: 0.1226, Batch time: 0.3066
Validate: 100%|█████████████████████████████| 2509/2509 [11:02<00:00,  3.79it/s]
tensor([0.9683, 0.6884, 0.5065, 0.8593, 0.9514, 0.9509, 0.9733, 0.8548, 0.6620,
        0.7969, 0.8229, 0.8797, 0.6875, 0.7414, 0.7236, 0.9312, 0.8845, 0.8289,
        0.8837, 0.8008, 0.9642, 0.6833, 0.7798, 0.8113, 0.8210, 0.9223, 0.7619,
        0.7920, 0.8993, 0.8407, 0.9823, 0.8468, 0.7948, 0.9328, 0.9934, 0.4189,
        0.5576, 0.9336, 0.7857, 0.8903, 0.9723, 0.6774, 0.8888, 0.8011, 0.9376,
        0.8748, 0.8167, 0.8697, 0.6842, 0.9909, 0.9427, 0.7009, 0.7513, 0.7899,
        0.7540, 0.6388, 0.9548, 0.8976, 0.9712, 0.9460, 0.8547, 0.6237, 0.8705,
        0.7839, 0.7216, 0.9451, 0.8696, 0.9897, 0.8517, 0.3165, 0.9744, 0.7092,
        0.8558, 0.9598, 0.7434, 0.8802, 0.8586, 0.7887, 0.7746, 0.9932])
* Test
Loss: 0.0414	 mAP: 0.8254	Data_time: 0.1508	 Batch_time: 0.2639
OP: 0.870	 OR: 0.726	 OF1: 0.792	CP: 0.870	 CR: 0.687	 CF1: 0.768
OP_3: 0.909	 OR_3: 0.647	 OF1_3: 0.756	CP_3: 0.902	 CR_3: 0.615	 CF1_3: 0.731
Save checkpoint to /workspace/cluster/HDD/azuma/Others/github/ML_DL_Notebook/MultiLabel_Image_Recognition/ADD-GCN/results/checkpoint.pth
Save results to /workspace/cluster/HDD/azuma/Others/github/ML_DL_Notebook/MultiLabel_Image_Recognition/ADD-GCN/results/results.csv
 * best mAP=0.8254
Lr: [0.005 0.05 ]
2025-09-23 10:28:25, 13 Epoch, 0 Iter, Loss: 0.0354, Data time: 1.1270, Batch time: 1.3122
2025-09-23 10:29:39, 13 Epoch, 200 Iter, Loss: 0.0537, Data time: 0.1023, Batch time: 0.2062
2025-09-23 10:30:55, 13 Epoch, 400 Iter, Loss: 0.0226, Data time: 0.0279, Batch time: 0.1332
2025-09-23 10:32:10, 13 Epoch, 600 Iter, Loss: 0.0486, Data time: 0.0362, Batch time: 0.2307
2025-09-23 10:33:25, 13 Epoch, 800 Iter, Loss: 0.0484, Data time: 0.1195, Batch time: 0.3205
2025-09-23 10:34:39, 13 Epoch, 1000 Iter, Loss: 0.0608, Data time: 0.1696, Batch time: 0.2997
2025-09-23 10:35:52, 13 Epoch, 1200 Iter, Loss: 0.0481, Data time: 0.0827, Batch time: 0.2041
2025-09-23 10:37:07, 13 Epoch, 1400 Iter, Loss: 0.0356, Data time: 0.1004, Batch time: 0.2300
2025-09-23 10:38:22, 13 Epoch, 1600 Iter, Loss: 0.0526, Data time: 0.0332, Batch time: 0.2033
2025-09-23 10:39:37, 13 Epoch, 1800 Iter, Loss: 0.0279, Data time: 0.0385, Batch time: 0.2290
2025-09-23 10:40:53, 13 Epoch, 2000 Iter, Loss: 0.0524, Data time: 0.0139, Batch time: 0.1329
2025-09-23 10:42:08, 13 Epoch, 2200 Iter, Loss: 0.0581, Data time: 0.1959, Batch time: 0.2954
2025-09-23 10:43:24, 13 Epoch, 2400 Iter, Loss: 0.0431, Data time: 0.1294, Batch time: 0.3272
2025-09-23 10:44:40, 13 Epoch, 2600 Iter, Loss: 0.0631, Data time: 0.1275, Batch time: 0.3177
2025-09-23 10:45:56, 13 Epoch, 2800 Iter, Loss: 0.0646, Data time: 0.0171, Batch time: 0.1292
2025-09-23 10:47:11, 13 Epoch, 3000 Iter, Loss: 0.0423, Data time: 0.0908, Batch time: 0.1999
2025-09-23 10:48:27, 13 Epoch, 3200 Iter, Loss: 0.0360, Data time: 0.0437, Batch time: 0.2296
2025-09-23 10:49:42, 13 Epoch, 3400 Iter, Loss: 0.0383, Data time: 0.0164, Batch time: 0.1293
2025-09-23 10:50:58, 13 Epoch, 3600 Iter, Loss: 0.0524, Data time: 0.0331, Batch time: 0.2028
2025-09-23 10:52:14, 13 Epoch, 3800 Iter, Loss: 0.0457, Data time: 0.0439, Batch time: 0.2324
2025-09-23 10:53:29, 13 Epoch, 4000 Iter, Loss: 0.0474, Data time: 0.0905, Batch time: 0.1999
2025-09-23 10:54:44, 13 Epoch, 4200 Iter, Loss: 0.0499, Data time: 0.1714, Batch time: 0.2982
2025-09-23 10:55:59, 13 Epoch, 4400 Iter, Loss: 0.0396, Data time: 0.0471, Batch time: 0.2277
2025-09-23 10:57:15, 13 Epoch, 4600 Iter, Loss: 0.0579, Data time: 0.1137, Batch time: 0.2278
2025-09-23 10:58:31, 13 Epoch, 4800 Iter, Loss: 0.0519, Data time: 0.0167, Batch time: 0.1286
2025-09-23 10:59:47, 13 Epoch, 5000 Iter, Loss: 0.0281, Data time: 0.0355, Batch time: 0.2297
Validate: 100%|█████████████████████████████| 2509/2509 [11:01<00:00,  3.79it/s]
tensor([0.9676, 0.6848, 0.5208, 0.8607, 0.9466, 0.9536, 0.9732, 0.8674, 0.6555,
        0.7883, 0.8108, 0.8891, 0.6826, 0.7390, 0.7502, 0.9324, 0.8595, 0.8354,
        0.8820, 0.7922, 0.9606, 0.6662, 0.7806, 0.8175, 0.8036, 0.9211, 0.7649,
        0.7880, 0.8959, 0.8381, 0.9822, 0.8529, 0.7834, 0.9355, 0.9953, 0.4247,
        0.5560, 0.9336, 0.8002, 0.8817, 0.9693, 0.6736, 0.8892, 0.7905, 0.9362,
        0.8768, 0.8175, 0.8674, 0.6952, 0.9899, 0.9434, 0.6956, 0.7691, 0.7804,
        0.7593, 0.6382, 0.9544, 0.8878, 0.9670, 0.9430, 0.8660, 0.6451, 0.8727,
        0.7833, 0.7155, 0.9472, 0.8745, 0.9904, 0.8313, 0.3531, 0.9770, 0.6963,
        0.8594, 0.9638, 0.7448, 0.8776, 0.8599, 0.7947, 0.7789, 0.9945])
* Test
Loss: 0.0426	 mAP: 0.8255	Data_time: 0.1505	 Batch_time: 0.2635
OP: 0.868	 OR: 0.717	 OF1: 0.785	CP: 0.853	 CR: 0.692	 CF1: 0.764
OP_3: 0.900	 OR_3: 0.645	 OF1_3: 0.752	CP_3: 0.884	 CR_3: 0.627	 CF1_3: 0.734
Save checkpoint to /workspace/cluster/HDD/azuma/Others/github/ML_DL_Notebook/MultiLabel_Image_Recognition/ADD-GCN/results/checkpoint.pth
Save results to /workspace/cluster/HDD/azuma/Others/github/ML_DL_Notebook/MultiLabel_Image_Recognition/ADD-GCN/results/results.csv
 * best mAP=0.8255
Lr: [0.005 0.05 ]
2025-09-23 11:13:19, 14 Epoch, 0 Iter, Loss: 0.0274, Data time: 1.4284, Batch time: 1.5651
2025-09-23 11:14:42, 14 Epoch, 200 Iter, Loss: 0.0522, Data time: 0.0445, Batch time: 0.2297
2025-09-23 11:16:04, 14 Epoch, 400 Iter, Loss: 0.0441, Data time: 0.1074, Batch time: 0.3048
2025-09-23 11:17:27, 14 Epoch, 600 Iter, Loss: 0.0330, Data time: 0.0470, Batch time: 0.2309
2025-09-23 11:18:49, 14 Epoch, 800 Iter, Loss: 0.0360, Data time: 0.0425, Batch time: 0.2279
2025-09-23 11:20:12, 14 Epoch, 1000 Iter, Loss: 0.0515, Data time: 0.1951, Batch time: 0.4705
2025-09-23 11:21:35, 14 Epoch, 1200 Iter, Loss: 0.0350, Data time: 0.1020, Batch time: 0.2212
2025-09-23 11:22:56, 14 Epoch, 1400 Iter, Loss: 0.0574, Data time: 0.2211, Batch time: 0.4308
2025-09-23 11:24:17, 14 Epoch, 1600 Iter, Loss: 0.0487, Data time: 0.1061, Batch time: 0.3084
ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
Traceback (most recent call last):
  File "/workspace/cluster/HDD/azuma/Others/github/ADD-GCN/main.py", line 70, in <module>
    main(args)
  File "/workspace/cluster/HDD/azuma/Others/github/ADD-GCN/main.py", line 64, in main
    trainer.train()
  File "/workspace/cluster/HDD/azuma/Others/github/ADD-GCN/trainer.py", line 149, in train
    self.run_iteration(self.train_loader, is_train=True)
  File "/workspace/cluster/HDD/azuma/Others/github/ADD-GCN/trainer.py", line 207, in run_iteration
    self.meters['ap_meter'].add(outputs.data, labels.data, data['name'])
  File "/workspace/cluster/HDD/azuma/Others/github/ADD-GCN/util.py", line 107, in add
    self.scores.narrow(0, offset, output.size(0)).copy_(output)
  File "/opt/250102_test_env/lib/python3.11/site-packages/torch/utils/data/_utils/signal_handling.py", line 73, in handler
    _error_if_any_worker_fails()
RuntimeError: DataLoader worker (pid 1226069) is killed by signal: Bus error. It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.
"""

# %%
