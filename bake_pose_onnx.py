# coding: utf-8
# Bake the 62-d param mean/std denormalization into the TDDFA regressor ONNX.
# Output graph takes a normalized 1x3xSIZExSIZE crop and returns the absolute
# 62-d param vector, so callers (e.g. demo_pose_only.py) need no pickle.

import argparse
import os.path as osp

import torch
import torch.nn as nn
import yaml

import models
from utils.io import _load
from utils.tddfa_util import load_model


class PoseBakedTDDFA(nn.Module):
    def __init__(self, regressor: nn.Module, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.regressor = regressor
        self.register_buffer('param_mean', mean.view(1, -1))
        self.register_buffer('param_std', std.view(1, -1))

    def forward(self, x):
        return self.regressor(x) * self.param_std + self.param_mean


def bake(config_fp: str, mean_std_fp: str, out_fp: str):
    cfg = yaml.safe_load(open(config_fp))
    size = cfg.get('size', 120)

    regressor = getattr(models, cfg['arch'])(
        num_classes=cfg.get('num_params', 62),
        widen_factor=cfg.get('widen_factor', 1),
        size=size,
        mode=cfg.get('mode', 'small'),
    )
    regressor = load_model(regressor, cfg['checkpoint_fp'])
    regressor.eval()

    stats = _load(mean_std_fp)
    mean = torch.from_numpy(stats['mean']).float()
    std = torch.from_numpy(stats['std']).float()
    if mean.numel() != cfg.get('num_params', 62):
        raise ValueError(
            f'mean has {mean.numel()} entries, expected {cfg.get("num_params", 62)}'
        )

    baked = PoseBakedTDDFA(regressor, mean, std).eval()

    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(
        baked,
        (dummy,),
        out_fp,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        do_constant_folding=True,
        dynamo=False,
    )
    print(f'Baked {cfg["checkpoint_fp"]} + {mean_std_fp} -> {out_fp}')


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Bake param mean/std into a TDDFA regressor ONNX'
    )
    parser.add_argument('-c', '--config', default='configs/mb1_120x120.yml',
                        help='YAML with arch / widen_factor / size / num_params / checkpoint_fp')
    parser.add_argument('-s', '--mean_std',
                        default='configs/param_mean_std_62d_120x120.pkl',
                        help='Pickle with {mean, std} for the 62-d params')
    parser.add_argument('-o', '--output', default=None,
                        help='Output ONNX path (default: <checkpoint_stem>_pose.onnx in weights/)')
    args = parser.parse_args(argv)

    if args.output is None:
        cfg = yaml.safe_load(open(args.config))
        stem = osp.splitext(osp.basename(cfg['checkpoint_fp']))[0]
        args.output = osp.join(osp.dirname(cfg['checkpoint_fp']), f'{stem}_pose.onnx')

    bake(args.config, args.mean_std, args.output)


if __name__ == '__main__':
    main()
