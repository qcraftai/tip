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

import cv2
import matplotlib as mpl
import numpy as np
import torch
from pyquaternion import Quaternion

mpl.use("Agg")
import os.path as osp
import pathlib

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns


def dir_exist(path, warning=False):
    if osp.isdir(path):
        return True
    if warning:
        print("I cannot find dir " + path)
    return False


def mkdir_safe(dir, warning=False):
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
    if not dir_exist(dir):
        return False
    return True


def get_corners(box, lw):
    l, w = lw
    simple_box = np.array(
        [
            [-l / 2.0, -w / 2.0],
            [l / 2.0, -w / 2.0],
            [l / 2.0, w / 2.0],
            [-l / 2.0, w / 2.0],
        ]
    )
    h = np.arctan2(box[3], box[2])
    rot = get_rot(h)
    simple_box = np.dot(simple_box, rot)
    simple_box += box[:2]
    return simple_box


def objects2frame(history, center, toworld=False):
    """A sphagetti function that converts from global
    coordinates to "center" coordinates or the inverse.
    It has no for loops but works on batchs.
    """
    N, A, B = history.shape
    theta = np.arctan2(center[3], center[2])
    if not toworld:
        newloc = history[:, :, :2] - center[:2].reshape((1, 1, 2))
        rot = get_rot(theta).T
        newh = np.arctan2(history[:, :, 3], history[:, :, 2]) - theta
        newloc = np.dot(newloc.reshape((N * A, 2)), rot).reshape((N, A, 2))
    else:
        rot = get_rot(theta)
        newh = np.arctan2(history[:, :, 3], history[:, :, 2]) + theta
        newloc = np.dot(history[:, :, :2].reshape((N * A, 2)), rot).reshape((N, A, 2))
    newh = np.stack((np.cos(newh), np.sin(newh)), 2)
    if toworld:
        newloc += center[:2]
    return np.append(newloc, newh, axis=2)


def get_rot(h):
    return np.array(
        [
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ]
    )


def raster_render(lmap, centerlw, lobjs, lws, nx, ny, layer_names, line_names, bx, dx):
    # draw both road layers vin one channel
    road_img = np.zeros((nx, ny))
    for layer_name in layer_names:
        for poly in lmap[layer_name]:
            # draw the lines
            pts = np.round((poly - bx[:2] + dx[:2] / 2.0) / dx[:2]).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(road_img, [pts], 1.0)

    def draw_lane(layer_name, img):
        for poly in lmap[layer_name]:
            pts = np.round((poly - bx[:2] + dx[:2] / 2.0) / dx[:2]).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.polylines(img, [pts], isClosed=False, color=1.0)
        return img

    road_div_img = np.zeros((nx, ny))
    draw_lane("road_divider", road_div_img)
    lane_div_img = np.zeros((nx, ny))
    draw_lane("lane_divider", lane_div_img)

    obj_img = np.zeros((nx, ny))
    for box, lw in zip(lobjs, lws):
        pts = get_corners(box, lw)
        # draw the box
        pts = np.round((pts - bx[:2] + dx[:2] / 2.0) / dx[:2]).astype(np.int32)
        pts[:, [1, 0]] = pts[:, [0, 1]]
        cv2.fillPoly(obj_img, [pts], 1.0)

    center_img = np.zeros((nx, ny))
    pts = get_corners([0.0, 0.0, 1.0, 0.0], centerlw)
    pts = np.round((pts - bx[:2] + dx[:2] / 2.0) / dx[:2]).astype(np.int32)
    pts[:, [1, 0]] = pts[:, [0, 1]]
    cv2.fillPoly(center_img, [pts], 1.0)

    return np.stack([road_img, road_div_img, lane_div_img, obj_img, center_img])


def get_grid(point_cloud_range, voxel_size):
    lower = np.array(point_cloud_range[: (len(point_cloud_range) // 2)])
    upper = np.array(point_cloud_range[(len(point_cloud_range) // 2) :])

    dx = np.array(voxel_size)
    bx = lower + dx / 2.0
    nx = ((upper - lower) / dx).astype(int)

    return dx, bx, nx


def make_rgba(probs, color):
    H, W = probs.shape
    return np.stack(
        (
            np.full((H, W), color[0]),
            np.full((H, W), color[1]),
            np.full((H, W), color[2]),
            probs,
        ),
        2,
    )


render_cm = dict(
    road=(0.85, 0.85, 0.85),
    road_div=(159.0 / 255.0, 0.0, 1.0),
    lane_div=(0.0, 0.0, 0.6),
    object=(0.0, 0.0, 0.0),
    ego=(0.5, 0.8, 0.5),
    fp=(1.0, 0.0, 0.0),
    fn=mpl.colors.to_rgb("darkgoldenrod"),
)


def render_observation(x, render_cm=render_cm):
    # road
    showimg = make_rgba(x[0].numpy().T, render_cm["road"])
    plt.imshow(showimg, origin="lower")

    # road div
    showimg = make_rgba(x[1].numpy().T, render_cm["road_div"])
    plt.imshow(showimg, origin="lower")

    # lane div
    showimg = make_rgba(x[2].numpy().T, render_cm["lane_div"])
    plt.imshow(showimg, origin="lower")

    # objects
    showimg = make_rgba(x[3].numpy().T, render_cm["object"])
    plt.imshow(showimg, origin="lower")

    # ego
    showimg = make_rgba(x[4].numpy().T, render_cm["ego"])
    plt.imshow(showimg, origin="lower")
    plt.grid(b=None)
    plt.xticks([])
    plt.yticks([])


def get_heatmap_colour(hm_palette_name="viridis", num_colour=12):
    hm_palette = sns.color_palette(hm_palette_name, n_colors=num_colour)
    hm_colours = [hm_palette[i] for i in range(0, num_colour)]
    return hm_colours


def plot_heatmap(heat, masks, colours, ts=12):
    plot_heat = heat.clone()
    plot_heat[~masks] = 0
    for ti in range(ts):
        flat = plot_heat[ti].view(-1)
        ixes = flat.topk(20).indices
        flat[ixes] = 1
        flat = flat.view(plot_heat.shape[1], plot_heat.shape[2])
        showimg = make_rgba(np.clip(flat.numpy().T, 0, 1), colours[ti])
        plt.imshow(showimg, origin="lower")


def analyze_plot(
    gtxs,
    predxs,
    gtdist_sig,
    preddist_sig,
    masks,
    scores=None,
    output_dir="./",
    fontsize=36,
):
    hm_colours = get_heatmap_colour()

    for i, (gtx, predx, gtsig, predsig) in enumerate(
        zip(gtxs, predxs, gtdist_sig, preddist_sig)
    ):
        fig = plt.figure(figsize=(35, 7), dpi=120)

        gs = mpl.gridspec.GridSpec(
            1,
            5,
            figure=fig,
            width_ratios=[1, 1, 1, 1, 1],
            height_ratios=[1],
            left=0.01,
            bottom=0.01,
            right=0.99,
            top=0.99,
            wspace=0,
            hspace=0,
        )

        gs_list = [gs[0, i] for i in range(5)]

        # GT
        ax = plt.subplot(gs_list[0])
        render_observation(gtx)
        ax.annotate(
            "Ground Truth", fontsize=fontsize, xy=(0.05, 0.05), xycoords="axes fraction"
        )
        plt.legend(
            handles=[
                mpatches.Patch(color=render_cm["road"], label="road"),
                mpatches.Patch(color=render_cm["object"], label="object"),
                mpatches.Patch(color=render_cm["ego"], label="AV"),
                mpatches.Patch(color=render_cm["road_div"], label="road divider"),
                mpatches.Patch(color=render_cm["lane_div"], label="lane boundary"),
            ],
            loc="upper left",
            fontsize=32,
            ncol=2,
            columnspacing=1.0,
            handlelength=0.5,
            handletextpad=0.5,
        )

        # detections
        ax = plt.subplot(gs_list[1])
        render_observation(predx)
        ax.annotate(
            "Detections", fontsize=fontsize, xy=(0.05, 0.05), xycoords="axes fraction"
        )

        # difference
        ax = plt.subplot(gs_list[2])
        new_obs = gtx.clone()
        new_obs[3] = 0
        render_observation(new_obs)
        showimg = make_rgba(
            np.clip((-gtx[3] + predx[3]).numpy().T, 0, 1), render_cm["fp"]
        )
        plt.imshow(showimg, origin="lower")
        showimg = make_rgba(
            np.clip((gtx[3] - predx[3]).numpy().T, 0, 1), render_cm["fn"]
        )
        plt.imshow(showimg, origin="lower")
        plt.legend(
            handles=[
                mpatches.Patch(color=render_cm["fp"], label="False Positive"),
                mpatches.Patch(color=render_cm["fn"], label="False Negative"),
            ],
            loc="upper right",
            fontsize=32,
            columnspacing=1.0,
            handlelength=0.5,
            handletextpad=0.5,
        )
        if scores is not None:
            mpl.rcParams.update(mpl.rcParamsDefault)
            ax.annotate(
                f"TIP = {scores[i]:.3f}",
                xy=(0.05, 0.05),
                xycoords="axes fraction",
                fontsize=24,
            )

        # behaviour: GT
        ax = plt.subplot(gs_list[3])
        plot_heatmap(gtsig, masks, colours=hm_colours)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.annotate(
            r"$p(a|S_{gt})$",
            fontsize=fontsize,
            xy=(0.05, 0.08),
            xycoords="axes fraction",
        )

        # behaviour: detections
        ax = plt.subplot(gs_list[4])
        plot_heatmap(predsig, masks, colours=hm_colours)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.annotate(
            r"$p(a|S_{detection})$",
            fontsize=fontsize,
            xy=(0.05, 0.08),
            xycoords="axes fraction",
        )

        imname = osp.join(output_dir, f"TIP_worst{i:04}.png")
        print("saving", imname)
        plt.savefig(imname)
        plt.close(fig)


def samp2ego(samp, nusc):
    egopose = nusc.get(
        "ego_pose",
        nusc.get("sample_data", samp["data"]["LIDAR_TOP"])["ego_pose_token"],
    )
    rot = Quaternion(egopose["rotation"]).rotation_matrix
    rot = np.arctan2(rot[1, 0], rot[0, 0])
    return {
        "x": egopose["translation"][0],
        "y": egopose["translation"][1],
        "hcos": np.cos(rot),
        "hsin": np.sin(rot),
        "l": 4.084,
        "w": 1.73,
    }


def samp2mapname(samp, nusc):
    scene = nusc.get("scene", samp["scene_token"])
    log = nusc.get("log", scene["log_token"])
    return log["location"]


def get_local_map(nmap, center, stretch, layer_names, line_names):
    # need to get the map here...
    box_coords = (
        center[0] - stretch,
        center[1] - stretch,
        center[0] + stretch,
        center[1] + stretch,
    )

    polys = {}

    # polygons
    records_in_patch = nmap.get_records_in_patch(
        box_coords, layer_names=layer_names, mode="intersect"
    )
    for layer_name in layer_names:
        polys[layer_name] = []
        for token in records_in_patch[layer_name]:
            poly_record = nmap.get(layer_name, token)
            if layer_name == "drivable_area":
                polygon_tokens = poly_record["polygon_tokens"]
            else:
                polygon_tokens = [poly_record["polygon_token"]]

            for polygon_token in polygon_tokens:
                polygon = nmap.extract_polygon(polygon_token)
                polys[layer_name].append(np.array(polygon.exterior.xy).T)

    # lines
    for layer_name in line_names:
        polys[layer_name] = []
        for record in getattr(nmap, layer_name):
            token = record["token"]

            line = nmap.extract_line(record["line_token"])
            if line.is_empty:  # Skip lines without nodes
                continue
            xs, ys = line.xy

            polys[layer_name].append(np.array([xs, ys]).T)

    # convert to local coordinates in place
    rot = get_rot(np.arctan2(center[3], center[2])).T
    for layer_name in polys:
        for rowi in range(len(polys[layer_name])):
            polys[layer_name][rowi] -= center[:2]
            polys[layer_name][rowi] = np.dot(polys[layer_name][rowi], rot)

    return polys


def get_other_objs(boxes, ego):
    objs = []
    lws = []
    for box in boxes:
        rot = Quaternion(box.rotation).rotation_matrix
        rot = np.arctan2(rot[1, 0], rot[0, 0])

        lws.append([box.size[1], box.size[0]])
        objs.append([box.translation[0], box.translation[1], np.cos(rot), np.sin(rot)])
    objs = np.array(objs)
    lws = np.array(lws)
    if len(objs) == 0:
        lobjs = np.zeros((0, 4))
    else:
        lobjs = objects2frame(
            objs[np.newaxis, :, :],
            ego,
        )[0]
    return lobjs, lws


def collect_x(
    sample_token,
    nusc,
    gt_boxes,
    pred_boxes,
    nusc_maps,
    stretch,
    layer_names,
    line_names,
    nx,
    ny,
    bx,
    dx,
):
    samp = nusc.get("sample", sample_token)

    # ego location
    ego = samp2ego(samp, nusc)

    # local map
    map_name = samp2mapname(samp, nusc)
    lmap = get_local_map(
        nusc_maps[map_name],
        [ego["x"], ego["y"], ego["hcos"], ego["hsin"]],
        stretch,
        layer_names,
        line_names,
    )

    # detections
    gtlobjs, gtlws = get_other_objs(
        gt_boxes[sample_token],
        np.array([ego["x"], ego["y"], ego["hcos"], ego["hsin"]]),
    )
    predlobjs, predlws = get_other_objs(
        pred_boxes[sample_token],
        np.array([ego["x"], ego["y"], ego["hcos"], ego["hsin"]]),
    )

    # render
    gtx = raster_render(
        lmap,
        [ego["l"], ego["w"]],
        gtlobjs,
        gtlws,
        nx,
        ny,
        layer_names,
        line_names,
        bx,
        dx,
    )
    predx = raster_render(
        lmap,
        [ego["l"], ego["w"]],
        predlobjs,
        predlws,
        nx,
        ny,
        layer_names,
        line_names,
        bx,
        dx,
    )

    return torch.Tensor(gtx), torch.Tensor(predx)


class EvalLoader(torch.utils.data.Dataset):
    def __init__(
        self,
        gt_boxes,
        pred_boxes,
        sample_tokens,
        nusc,
        nusc_maps,
        stretch,
        layer_names,
        line_names,
    ):
        self.dx, self.bx, (self.nx, self.ny) = get_grid(
            [-17.0, -38.5, 60.0, 38.5], [0.3, 0.3]
        )
        self.gt_boxes = gt_boxes
        self.pred_boxes = pred_boxes
        self.sample_tokens = sample_tokens
        self.nusc = nusc
        self.nusc_maps = nusc_maps
        self.stretch = stretch
        self.layer_names = layer_names
        self.line_names = line_names

    def __len__(self):
        return len(self.sample_tokens)

    def __getitem__(self, index):
        return collect_x(
            self.sample_tokens[index],
            self.nusc,
            self.gt_boxes,
            self.pred_boxes,
            self.nusc_maps,
            self.stretch,
            self.layer_names,
            self.line_names,
            self.nx,
            self.ny,
            self.bx,
            self.dx,
        )
