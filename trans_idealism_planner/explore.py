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

import json
import os
from os import path as osp

import matplotlib as mpl
import numpy as np
import torch
from nuscenes.eval.common.config import config_factory
from nuscenes.nuscenes import NuScenes
from tqdm import trange

mpl.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from trans_idealism_planner.ti_planner import get_tip

from .data import compile_data
from .models import compile_model
from .tools import (
    SimpleLoss,
    TIPEval,
    get_nusc_maps,
    get_scene2samp,
    plot_box,
    safe_forward,
)
from .utils import make_rgba, render_cm, render_observation


def false_neg_viz(
    version,
    model_path="./planner.pt",
    dataroot="/data/nuscenes",
    map_folder="/data/nuscenes/mini/",
    output_dir="fneg_demo",
    normalizing_factor=0.2,
    ts=12,
    ego_only=True,
    t_spacing=0.5,
    bsz=8,
    num_workers=10,
    flip_aug=False,
    dropout_p=0.0,
    mask_json="masks_trainval.json",
    pos_weight=10.0,
    loss_clip=True,
    gpuid=0,
):
    """Remove each true detection to measure the "importance" of each object."""
    device = torch.device(f"cuda:{gpuid}") if gpuid >= 0 else torch.device("cpu")
    print(f"using device: {device}")

    trainloader, valloader = compile_data(
        version,
        dataroot,
        map_folder,
        ego_only,
        t_spacing,
        bsz,
        num_workers,
        flip_aug,
    )

    model = compile_model(cin=5, cout=16, with_skip=True, dropout_p=dropout_p).to(
        device
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    loss_fn = SimpleLoss(mask_json, pos_weight, loss_clip, device, ts=ts)

    dataset = valloader.dataset

    get_metric = get_tip

    with torch.no_grad():
        for batchi in trange(len(dataset)):
            scene, name, t0 = dataset.ixes[batchi]
            lmap, centerlw, lobjs, lws, _ = dataset.get_state(scene, name, t0)

            num_objs = len(lobjs)
            scores = []
            if num_objs > 0:
                x = dataset.render(lmap, centerlw, lobjs, lws)
                x = torch.Tensor(x).unsqueeze(0)
                pred = model(x.to(device))
                pred = torch.cat([pred for _ in range(len(lobjs))])

                xs = []
                for drop_ix in range(len(lobjs)):
                    new_objs = [
                        obj for obji, obj in enumerate(lobjs) if obji != drop_ix
                    ]
                    new_lws = [lw for obji, lw in enumerate(lws) if obji != drop_ix]

                    x = dataset.render(lmap, centerlw, new_objs, new_lws)
                    xs.append(torch.Tensor(x))
                xs = torch.stack(xs)
                preds = model(xs.to(device))

                scores = get_metric(
                    masks=loss_fn.masks,
                    pred_xs_logits=preds,
                    gt_xs_logits=pred,
                )

            fig = plt.figure(figsize=(4, 4), dpi=120)
            gs = mpl.gridspec.GridSpec(
                1, 1, left=0, bottom=0, right=1, top=1, wspace=0, hspace=0
            )

            ax = plt.subplot(gs[0, 0])

            # plot map
            for layer_name in dataset.layer_names:
                for poly in lmap[layer_name]:
                    plt.fill(poly[:, 0], poly[:, 1], c=render_cm["road"])

            for poly in lmap["road_divider"]:
                plt.plot(poly[:, 0], poly[:, 1], c=render_cm["road_div"])
            for poly in lmap["lane_divider"]:
                plt.plot(poly[:, 0], poly[:, 1], c=render_cm["lane_div"])

            # plot objects
            for lobj, lw, score in zip(lobjs, lws, scores):
                plot_box(
                    lobj,
                    lw,
                    "r",
                    alpha=(score.abs() / normalizing_factor).clamp(0.1, 1).item(),
                )

            # plot ego
            plot_box([0.0, 0.0, 1.0, 0.0], [4.084, 1.73], render_cm["ego"])

            plt.legend(
                handles=[
                    mpatches.Patch(color=render_cm["road"], label="road"),
                    mpatches.Patch(color="r", label="object"),
                    mpatches.Patch(color=render_cm["ego"], label="AV"),
                    mpatches.Patch(color=render_cm["road_div"], label="road divider"),
                    mpatches.Patch(color=render_cm["lane_div"], label="lane boundary"),
                ],
                ncol=2,
                loc="upper left",
                fontsize=12,
            )

            plt.xlim((-17, 60))
            plt.ylim((-38.5, 38.5))
            ax.set_aspect("equal")
            plt.axis("off")

            imname = osp.join(output_dir, f"fneg{batchi:06}.png")
            plt.savefig(imname)
            plt.close(fig)


def false_pos_viz(
    version,
    model_path,
    dataroot="/data/nuscenes",
    map_folder="/data/nuscenes/mini/",
    output_dir="fpos_demo",
    normalizing_factor=0.2,
    ts=12,
    ego_only=True,
    t_spacing=0.5,
    bsz=8,
    num_workers=10,
    flip_aug=False,
    dropout_p=0.0,
    mask_json="masks_trainval.json",
    pos_weight=10.0,
    loss_clip=True,
    gpuid=0,
):
    """Add a false positive at each (x,y) position in a grid about the ego."""
    device = torch.device(f"cuda:{gpuid}") if gpuid >= 0 else torch.device("cpu")
    print(f"using device: {device}")

    trainloader, valloader = compile_data(
        version,
        dataroot,
        map_folder,
        ego_only,
        t_spacing,
        bsz,
        num_workers,
        flip_aug,
    )

    model = compile_model(cin=5, cout=16, with_skip=True, dropout_p=dropout_p).to(
        device
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    loss_fn = SimpleLoss(mask_json, pos_weight, loss_clip, device, ts=ts)

    plt.figure(figsize=(4, 4), dpi=120)
    gs = mpl.gridspec.GridSpec(1, 1)

    dataset = valloader.dataset

    plt.figure(figsize=(4, 4), dpi=120)
    gs = mpl.gridspec.GridSpec(
        1, 1, left=0, bottom=0, right=1, top=1, wspace=0, hspace=0
    )

    nx = 32
    ny = 32
    dxi = 8
    dxj = 4
    with torch.no_grad():
        for batchi in trange(len(dataset)):
            scene, name, t0 = dataset.ixes[batchi]
            lmap, centerlw, lobjs, lws, _ = dataset.get_state(scene, name, t0)
            x = dataset.render(lmap, centerlw, lobjs, lws)
            x = torch.Tensor(x).unsqueeze(0)
            pred = model(x.to(device))
            pred_sig = pred.sigmoid()

            xixes = torch.linspace(0, x.shape[2] - 1, nx).long()
            yixes = torch.linspace(0, x.shape[3] - 1, ny).long()
            xs = []
            for i in xixes:
                for j in yixes:
                    imgij = x.clone()
                    loweri = max(0, j - dxi)
                    upperi = j + dxi
                    lowerj = max(0, x.shape[2] - i - dxj)
                    upperj = x.shape[2] - i + dxj
                    imgij[0, 3, loweri:upperi, lowerj:upperj] = 1
                    xs.append(imgij)
            xs = torch.cat(xs)

            scores = safe_forward(model, xs, device, pred, pred_sig, loss_fn).abs()
            scores = scores.view(1, 1, nx, ny)
            up_scores = torch.nn.Upsample(
                size=(x.shape[2], x.shape[3]),
                mode="bilinear",
                align_corners=True,
            )(scores)
            up_scores = (up_scores / normalizing_factor).clamp(0.0, 1.0)

            plt.subplot(gs[0, 0])
            render_observation(x.squeeze(0))
            showimg = make_rgba(up_scores[0, 0], (0.5, 0, 0))
            plt.imshow(
                np.flip(showimg, 0),
                origin="lower",
            )

            plt.legend(
                handles=[
                    mpatches.Patch(color=render_cm["road"], label="road"),
                    mpatches.Patch(color=render_cm["object"], label="object"),
                    mpatches.Patch(color=render_cm["ego"], label="AV"),
                    mpatches.Patch(color=render_cm["road_div"], label="road divider"),
                    mpatches.Patch(color=render_cm["lane_div"], label="lane boundary"),
                ],
                ncol=2,
                loc="upper left",
                fontsize=12,
            )

            imname = osp.join(output_dir, f"fpos{batchi:06}.png")
            plt.savefig(imname)
            plt.clf()


def eval_test(
    version,
    eval_set,
    result_path,
    model_path="./planner.pt",
    dataroot="/data/nuscenes",
    map_folder="/data/nuscenes/mini/",
    output_dir="./res",
    ts=12,  # this is equal to 3s since the mask dt is 0.25s
    nworkers=8,
    plot_kextremes=5,
    gpuid=0,
    mask_json="./masks_trainval.json",
):
    """Evaluate detections with PKL / TIP."""
    nusc = NuScenes(
        version="v1.0-{}".format(version),
        dataroot=os.path.join(dataroot, version),
        verbose=True,
    )
    nusc_maps = get_nusc_maps(map_folder)
    cfg = config_factory("detection_cvpr_2019")
    device = torch.device(f"cuda:{gpuid}") if gpuid >= 0 else torch.device("cpu")
    print(f"using device: {device}")

    nusc_eval = TIPEval(
        nusc,
        config=cfg,
        result_path=result_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=True,
    )
    info = nusc_eval.evaluate(
        nusc_maps,
        device,
        ts=ts,
        nworkers=nworkers,
        plot_kextremes=plot_kextremes,
        model_path=model_path,
        mask_json=mask_json,
    )
    print({k: v for k, v in info.items() if k != "full"})

    output_stat_path = osp.join(output_dir, "planner_metric_tip.json")
    print("saving results to", output_stat_path)
    with open(output_stat_path, "w") as writer:
        json.dump(info, writer)


def score_distribution_plot(
    version,
    score_result_path,
    plot_name="dist.png",
    ntrials=10,
    dataroot="/data/nuscenes",
):
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    nusc = NuScenes(
        version="v1.0-{}".format(version),
        dataroot=os.path.join(dataroot, version),
        verbose=True,
    )

    with open(score_result_path, "r") as reader:
        info = json.load(reader)

    reverse = True
    if reverse:
        info["full"] = {k: -np.abs(v) for k, v in info["full"].items()}
    vals = [val for val in info["full"].values()]
    info["min"] = np.min(vals)
    info["max"] = np.max(vals)
    info["mean"] = np.mean(vals)
    info["median"] = np.median(vals)
    info["std"] = np.mean(vals)

    if np.min(vals) >= 0:
        range_min = 0
        range_max = 2 * info["mean"]
    else:
        range_min = 2 * info["mean"]
        range_max = 0

    # need to get scene->samples (ordered by timestamp)
    scene2samp = get_scene2samp(nusc, info["full"])

    fig = plt.figure(figsize=(15, 4), dpi=120)
    gs = mpl.gridspec.GridSpec(1, 3)

    # histogram
    ax = plt.subplot(gs[0, 0])
    plt.hist(
        vals,
        bins=40,
        range=(range_min, range_max),
        label="Samples",
        color="b",
    )
    plt.axvline(
        info["mean"],
        label=f"Overall Mean score: {info['mean']:.02f}",
        c="g",
        linestyle="dashed",
    )
    plt.axvline(
        info["median"],
        label=f"Overall Median score: {info['median']:.02f}",
        c="r",
        linestyle="dashed",
    )
    plt.xlabel("TIP")
    plt.ylabel("N samples")
    plt.legend()
    plt.xlim((range_min, range_max))

    # score vs #sample
    ax_mean = plt.subplot(gs[0, 1])
    for tr in range(ntrials):
        # random order
        vals = np.random.permutation(vals)
        xs = np.arange(1, len(vals) + 1)
        # mean
        ys = [np.mean(vals[:i]) for i in xs]
        plt.sca(ax_mean)
        if tr == 0:
            plt.plot(xs, ys, "b", alpha=0.5, label="Moving endpoint mean")
            plt.axhline(
                info["mean"],
                label=f"Overall Mean score: {info['mean']:.02f}",
                c="g",
                linestyle="dashed",
            )
            plt.legend()
            plt.xlabel("N samples")
        else:
            plt.plot(xs, ys, "b", alpha=0.5)
        # median
        ys = [np.median(vals[:i]) for i in xs]
        # plt.sca(ax_median)
        if tr == 0:
            plt.plot(xs, ys, "r", alpha=0.5, label="Moving endpoint median")
            plt.axhline(
                info["median"],
                label=f"Overall Median score: {info['median']:.02f}",
                c="k",
                linestyle="dashed",
            )
            plt.ylabel("score")
            plt.legend()
            plt.xlabel("N samples")
        else:
            plt.plot(xs, ys, "r", alpha=0.5)
    ax_mean.set_ylim((range_min, range_max))

    ax = plt.subplot(gs[0, 2])
    medians = []
    means = []
    for nperscene in range(1, 25):
        tokens = []
        for scene in scene2samp:
            ixes = np.round(np.linspace(0, len(scene2samp[scene]) - 1, nperscene))
            tokens.extend([scene2samp[scene][int(ix)] for ix in ixes])
        medians.append(
            (nperscene, np.median([info["full"][token] for token in tokens]))
        )
        means.append((nperscene, np.mean([info["full"][token] for token in tokens])))
    plt.plot(
        [row[0] for row in means],
        [row[1] for row in means],
        "b",
        label="Mean score",
    )
    plt.plot(
        [row[0] for row in medians],
        [row[1] for row in medians],
        "r",
        label="Median score",
    )
    plt.axhline(
        info["median"],
        label=f"Overall Median score: {info['median']:.02f}",
        c="r",
        linestyle="dashed",
    )
    plt.axhline(
        info["mean"],
        label=f"Overall Mean score: {info['mean']:.02f}",
        c="b",
        linestyle="dashed",
    )
    plt.legend()
    plt.xlabel("N samples per scene")
    plt.ylabel("score")
    ax.set_ylim((range_min, range_max))

    plt.tight_layout()
    print("saving", plot_name)
    plt.savefig(plot_name)
    plt.close(fig)
