# Copyright (C) 2023 QCraft, Inc., Wei-Xin Li, Xiaodong Yang.
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

import matplotlib as mpl
import numpy as np
import torch
from tqdm import tqdm

mpl.use("Agg")
import os
import os.path as osp

from .models import compile_model
from .utils import EvalLoader, analyze_plot


def load_model(device, model_path, verbose=True):
    model = compile_model(cin=5, cout=16, with_skip=True, dropout_p=0.0).to(device)
    if not osp.isfile(model_path):
        print(f"downloading model weights to location {model_path}...")
        cmd = f"wget --quiet --no-check-certificate 'https://docs.google.com/uc?export=download&id=1feEIUjYSNWkl_b5SUkmPZ_-JAj3licJ9' -O {model_path}"
        print(f"running {cmd}")
        os.system(cmd)
    if verbose:
        print(f"using model weights {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.to(device)
    model.eval()

    return model


def load_masks(device, mask_json, ts=3, verbose=True):
    if not osp.isfile(mask_json):
        print(f"downloading model masks to location {mask_json}...")
        cmd = f"wget --quiet --no-check-certificate 'https://docs.google.com/uc?export=download&id=13M1xj9MkGo583ok9z8EkjQKSV8I2nWWF' -O {mask_json}"
        print(f"running {cmd}")
        os.system(cmd)
    if verbose:
        print(f"using location masks {mask_json}")
    with open(mask_json, "r") as reader:
        masks = (torch.Tensor(json.load(reader)) == 1).to(device)

    T_masks = masks.size(dim=1)
    assert ts <= T_masks, f"ts = {ts} cannot be larger than T_masks = {T_masks}!"
    if ts < T_masks:
        masks[ts:, :] = False

    return masks


def calculate_tip(
    gt_boxes,
    pred_boxes,
    sample_tokens,
    nusc,
    nusc_maps,
    device,
    nworkers,
    bsz=128,
    ts=12,
    plot_kextremes=0,
    output_dir=None,
    verbose=True,
    model_path="./planner.pt",
    mask_json="./masks_trainval.json",
):
    """Computes the TIP https://arxiv.org/abs/2306.07276.

    It is designed to consume boxes in the format from
    nuscenes.eval.detection.evaluate.DetectionEval.
    Args:
            gt_boxes (EvalBoxes): Ground truth objects
            pred_boxes (EvalBoxes): Predicted objects
            sample_tokens List[str]: timestamps to be evaluated
            nusc (NuScenes): parser object provided by nuscenes-devkit
            nusc_maps (dict): maps map names to NuScenesMap objects
            device (torch.device): device for running forward pass
            nworkers (int): number of workers for dataloader
            bsz (int): batch size for dataloader
            plot_kextremes (int): number of examples to plot
            verbose (bool): print or not
            model_path (str): File path to model weights.
                             Will download if not found.
            mask_json (str): File path to trajectory masks.
                             Will download if not found.
    Returns:
            info (dict) : dictionary of PKL scores
    """

    # constants related to how the planner was trained
    layer_names = ["road_segment", "lane"]
    line_names = ["road_divider", "lane_divider"]
    stretch = 70.0

    # load planner
    model = load_model(device=device, model_path=model_path, verbose=verbose)

    # load masks
    masks = load_masks(device=device, mask_json=mask_json, ts=ts, verbose=verbose)

    dataset = EvalLoader(
        gt_boxes,
        pred_boxes,
        sample_tokens,
        nusc,
        nusc_maps,
        stretch,
        layer_names,
        line_names,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bsz,
        shuffle=False,
        num_workers=nworkers,
    )

    if verbose:
        print("calculating tip...")

    all_tips = []

    for gtxs, predxs in tqdm(dataloader):

        with torch.no_grad():
            gtdist = model(gtxs.to(device))
            preddist = model(predxs.to(device))

        # tip
        tips = get_tip(
            masks,
            preddist,
            gtdist,
            ts=ts,
            # use_prob=False,
            to_tensor=False,
        )
        all_tips.append(tips)

    all_tips = np.concatenate(all_tips, axis=0)
    all_tips = torch.from_numpy(all_tips)  # to conply with PKL's definition

    info = parse_result(
        all_tips,
        plot_kextremes,
        device,
        model,
        dataset,
        masks,
        sample_tokens,
        output_dir=output_dir,
        verbose=True,
    )

    return info


def parse_result(
    scores,
    plot_kextremes,
    device,
    model,
    dataset,
    masks,
    sample_tokens,
    output_dir=None,
    verbose=True,
):
    # plot k extremes
    if verbose:
        print(f"plotting {plot_kextremes} timestamps to {output_dir}...")
    if plot_kextremes > 0 and output_dir is not None:
        worst_ixes = scores.abs().topk(plot_kextremes).indices
        out = [dataset[i] for i in worst_ixes]
        gtxs, predxs = list(zip(*out))
        gtxs, predxs = torch.stack(gtxs), torch.stack(predxs)
        with torch.no_grad():
            gtdist = model(gtxs.to(device))
            gtdist_sig = gtdist.sigmoid()
            preddist = model(predxs.to(device))
        analyze_plot(
            gtxs,
            predxs,
            gtdist_sig.cpu(),
            preddist.sigmoid().cpu(),
            masks.cpu(),
            output_dir=output_dir,
            scores=scores[worst_ixes],
        )

    return {
        "min": scores.min().item(),
        "max": scores.max().item(),
        "mean": scores.mean().item(),
        "median": scores.median().item(),
        "std": scores.std().item(),
        "full": {tok: pk.item() for tok, pk in zip(sample_tokens, scores)},
        "top_worst": {i.item(): scores[i].item() for i in worst_ixes}
        if plot_kextremes > 0
        else None,
    }


def get_distribution_from_logits(logits):
    """
    logits shape: (D).
    """
    max_logits = np.max(logits)
    logits -= max_logits
    logit_exp = np.exp(logits)
    logit_exp_sum = np.sum(logit_exp)
    return logit_exp / logit_exp_sum


def get_xs_distribution(mask, xs_logits):
    xs_logit = xs_logits[mask]
    return get_distribution_from_logits(xs_logit)


def get_tip(masks, pred_xs_logits, gt_xs_logits, ts=12, use_prob=True, to_tensor=True):
    """
    masks are of the shape (16, 256, 256),
    logits are of the shape (N, 16, 256, 256).

    if use_prob, probability density is used as the utility function; otherwise
    the log density is used.
    """
    if torch.is_tensor(pred_xs_logits):
        pred_xs_logits = pred_xs_logits.cpu().detach().numpy()
    if torch.is_tensor(gt_xs_logits):
        gt_xs_logits = gt_xs_logits.cpu().detach().numpy()
    if torch.is_tensor(masks):
        masks = masks.cpu().detach().numpy()

    logit_shape = pred_xs_logits.shape

    assert np.all(
        logit_shape == gt_xs_logits.shape
    ), f"shapres are not identical: {logit_shape} != {gt_xs_logits.shape}!"

    num_frame = logit_shape[1]
    assert ts <= num_frame, f"no enough frames for evaluation {ts} > {num_frame}!"

    num_sample = logit_shape[0]
    tips = []  # of shape (#sample, ts)
    for i in range(num_sample):
        raw_scores = []
        for t in range(ts):
            mask = masks[t, :]
            pred_xs_prob = get_xs_distribution(
                mask, np.squeeze(pred_xs_logits[i, t, :, :])
            )
            # pred_xs_probs.append(pred_xs_prob)

            gt_xs_prob = get_xs_distribution(mask, np.squeeze(gt_xs_logits[i, t, :, :]))

            a_star_idx = np.unravel_index(gt_xs_prob.argmax(), gt_xs_prob.shape)

            xs_prob_diff = gt_xs_prob - pred_xs_prob
            a_hat_idx = np.unravel_index(xs_prob_diff.argmin(), xs_prob_diff.shape)

            raw_scores.append(
                dict(
                    EpU_Astar=gt_xs_prob[a_star_idx],
                    EqU_Astar=pred_xs_prob[a_star_idx],
                    EpU_Ahat=gt_xs_prob[a_hat_idx],
                    EqU_Ahat=pred_xs_prob[a_hat_idx],
                )
            )

        if not use_prob:
            raw_scores = [{k: np.log(v) for k, v in rs.items()} for rs in raw_scores]

        tips.append(
            [
                rs["EqU_Astar"] - rs["EqU_Ahat"] - rs["EpU_Astar"] + rs["EpU_Ahat"]
                for rs in raw_scores
            ]
        )

    tip_scores = np.sum(tips, axis=1)

    if to_tensor:
        tip_scores = torch.from_numpy(tip_scores)

    return tip_scores
