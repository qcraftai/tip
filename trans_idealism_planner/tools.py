# Modifications Copyright (C) 2023 QCraft, Inc., Wei-Xin Li, Xiaodong Yang.
#
# ------------------------------------------------------------------------
#
# Copyright 2020 NVIDIA CORPORATION, Jonah Philion, Amlan Kar, Sanja Fidler
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib as mpl
import numpy as np
import torch
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.map_expansion.map_api import NuScenesMap

mpl.use("Agg")
import matplotlib.pyplot as plt

from .ti_planner import calculate_tip, get_tip, load_masks
from .utils import get_corners, mkdir_safe


def get_nusc_maps(map_folder):
    nusc_maps = {
        map_name: NuScenesMap(dataroot=map_folder, map_name=map_name)
        for map_name in [
            "singapore-hollandvillage",
            "singapore-queenstown",
            "boston-seaport",
            "singapore-onenorth",
        ]
    }
    return nusc_maps


def get_scene2samp(nusc, full):
    scene2samp = {}
    for token in full:
        samp = nusc.get("sample", token)
        scene = nusc.get("scene", samp["scene_token"])["name"]
        if scene not in scene2samp:
            scene2samp[scene] = []
        scene2samp[scene].append(token)
    for scene in scene2samp:
        scene2samp[scene] = sorted(
            scene2samp[scene],
            key=lambda x: nusc.get("sample", x)["timestamp"],
        )
    return scene2samp


def plot_box(box, lw, color, alpha=0.8):
    l, w = lw
    h = np.arctan2(box[3], box[2])
    simple_box = get_corners(box, lw)

    arrow = np.array(
        [
            box[:2],
            box[:2] + l / 2.0 * np.array([np.cos(h), np.sin(h)]),
        ]
    )

    plt.fill(
        simple_box[:, 0],
        simple_box[:, 1],
        color=color,
        edgecolor="k",
        alpha=alpha,
    )
    plt.plot(arrow[:, 0], arrow[:, 1], "k")


class SimpleLoss(torch.nn.Module):
    def __init__(self, mask_json, pos_weight, loss_clip, device, ts=16):
        super(SimpleLoss, self).__init__()

        # with open(mask_json, "r") as reader:
        #     self.masks = (torch.Tensor(json.load(reader)) == 1).to(device)

        self.masks = load_masks(
            device=device,
            mask_json=mask_json,
            ts=ts,
        )

        self.pos_weight = pos_weight
        self.loss_clip = loss_clip
        self.use_mask = True

    def forward(self, pred, y):
        weight = torch.ones_like(pred)
        weight[y == 1] = self.pos_weight
        new_y = y.clone()
        if self.loss_clip:
            new_y[new_y == 0] = 0.01
        if self.use_mask:
            loss = F.binary_cross_entropy_with_logits(
                pred[:, self.masks],
                new_y[:, self.masks],
                weight[:, self.masks],
                reduction="mean",
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                pred,
                new_y,
                weight,
                reduction="mean",
            )
        return loss

    def accuracy(self, pred, y):
        """pred should have already been sigmoided, y is gt"""
        B, C, H, W = pred.shape
        new_pred = pred.clone()
        # hack to make sure argmax is inside the mask
        if self.use_mask:
            new_pred[:, self.masks] += 1.0
        K = 5
        correct = (
            new_pred.view(B, C, H * W).topk(K, 2).indices
            == y.view(B, C, H * W).max(2, keepdim=True).indices
        )
        final = correct.float().sum(2).sum(0)

        return final


def safe_forward(model, xs, device, pred, pred_sig, loss_fn, bsz=128):
    scores = []

    get_metric = get_tip

    with torch.no_grad():
        for i in range(0, xs.shape[0], bsz):
            x = xs[i : (i + bsz)]
            preds = model(x.to(device))
            tgt_sig = torch.cat([pred_sig for _ in range(preds.shape[0])])
            tgt = torch.cat([pred for _ in range(preds.shape[0])])

            scores.append(
                # (
                #     F.binary_cross_entropy_with_logits(
                #         preds[:, loss_fn.masks],
                #         tgt_sig[:, loss_fn.masks],
                #         reduction="none",
                #     )
                #     - F.binary_cross_entropy_with_logits(
                #         tgt[:, loss_fn.masks],
                #         tgt_sig[:, loss_fn.masks],
                #         reduction="none",
                #     )
                # ).sum(1)
                get_metric(
                    masks=loss_fn.masks,
                    pred_xs_logits=preds,
                    gt_xs_logits=tgt,
                )
            )
    return torch.cat(scores).cpu()


class MetricEval(DetectionEval):
    def __init__(self, *args, **kwargs):
        super(MetricEval, self).__init__(*args, **kwargs)
        self.metric_output_dir = self.output_dir
        mkdir_safe(self.metric_output_dir)


class TIPEval(MetricEval):
    def __init__(self, *args, **kwargs):
        super(TIPEval, self).__init__(*args, **kwargs)

    def evaluate(
        self,
        nusc_maps,
        device,
        nworkers,
        ts=12,  # this is equal to 3s since the mask dt is 0.25s
        bsz=128,
        plot_kextremes=0,
        model_path="./planner.pt",
        mask_json="./masks_trainval.json",
    ):
        return calculate_tip(
            self.gt_boxes,
            self.pred_boxes,
            self.sample_tokens,
            self.nusc,
            nusc_maps,
            device,
            nworkers,
            bsz=bsz,
            ts=ts,
            plot_kextremes=plot_kextremes,
            output_dir=self.metric_output_dir,
            verbose=self.verbose,
            model_path=model_path,
            mask_json=mask_json,
        )
