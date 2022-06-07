from dataclasses import dataclass, field
from typing import Any, ClassVar, List, Tuple
import argparse
import functools
import logging
import math

import torch
import numpy as np


@dataclass
class Base:
    name: str
    dataset: str

    head_index: int = field(default=None, init=False)
    base_stride: int = field(default=None, init=False)
    upsample_stride: int = field(default=1, init=False)

    @property
    def stride(self) -> int:
        if self.base_stride is None:
            return None
        return self.base_stride // self.upsample_stride

    @property
    def n_fields(self) -> int:
        raise NotImplementedError

class HeadNetwork(torch.nn.Module):
    """Base class for head networks.
    :param meta: head meta instance to configure this head network
    :param in_features: number of input features which should be equal to the
        base network's output features
    """
    def __init__(self, meta: Base, in_features: int):
        super().__init__()
        self.meta = meta
        self.in_features = in_features

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        """Command line interface (CLI) to extend argument parser."""

    @classmethod
    def configure(cls, args: argparse.Namespace):
        """Take the parsed argument parser output and configure class variables."""

    def forward(self, x):
        raise NotImplementedError

class CompositeField4(HeadNetwork):
    dropout_p = 0.0
    inplace_ops = True

    def __init__(self,
                 meta: Base,
                 in_features, *,
                 kernel_size=1, padding=0, dilation=1):
        super().__init__(meta, in_features)


        self.dropout = torch.nn.Dropout2d(p=self.dropout_p)

        # convolution
        self.n_components = 1 + meta.n_confidences + meta.n_vectors * 2 + meta.n_scales
        self.conv = torch.nn.Conv2d(
            in_features, meta.n_fields * self.n_components * (meta.upsample_stride ** 2),
            kernel_size, padding=padding, dilation=dilation,
        )

        # upsample
        assert meta.upsample_stride >= 1
        self.upsample_op = None
        if meta.upsample_stride > 1:
            self.upsample_op = torch.nn.PixelShuffle(meta.upsample_stride)

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('CompositeField4')
        group.add_argument('--cf4-dropout', default=cls.dropout_p, type=float,
                           help='[experimental] zeroing probability of feature in head input')
        assert cls.inplace_ops
        group.add_argument('--cf4-no-inplace-ops', dest='cf4_inplace_ops',
                           default=True, action='store_false',
                           help='alternative graph without inplace ops')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.dropout_p = args.cf4_dropout
        cls.inplace_ops = args.cf4_inplace_ops

    @property
    def sparse_task_parameters(self):
        return [self.conv.weight]

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.dropout(x)
        x = self.conv(x)
        # upscale
        if self.upsample_op is not None:
            x = self.upsample_op(x)
            low_cut = (self.meta.upsample_stride - 1) // 2
            high_cut = math.ceil((self.meta.upsample_stride - 1) / 2.0)
            if self.training:
                # negative axes not supported by ONNX TensorRT
                x = x[:, :, low_cut:-high_cut, low_cut:-high_cut]
            else:
                # the int() forces the tracer to use static shape
                x = x[:, :, low_cut:int(x.shape[2]) - high_cut, low_cut:int(x.shape[3]) - high_cut]

        # Extract some shape parameters once.
        # Convert to int so that shape is constant in ONNX export.
        x_size = x.size()
        batch_size = x_size[0]
        feature_height = int(x_size[2])
        feature_width = int(x_size[3])

        x = x.view(
            batch_size,
            self.meta.n_fields,
            self.n_components,
            feature_height,
            feature_width
        )

        if not self.training and self.inplace_ops:
            # classification
            classes_x = x[:, :, 1:1 + self.meta.n_confidences]
            torch.sigmoid_(classes_x)

            # regressions x: add index
            if self.meta.n_vectors > 0:
                index_field = index_field_torch((feature_height, feature_width), device=x.device)
                first_reg_feature = 1 + self.meta.n_confidences
                for i, do_offset in enumerate(self.meta.vector_offsets):
                    if not do_offset:
                        continue
                    reg_x = x[:, :, first_reg_feature + i * 2:first_reg_feature + (i + 1) * 2]
                    reg_x.add_(index_field)

            # scale
            first_scale_feature = 1 + self.meta.n_confidences + self.meta.n_vectors * 2
            scales_x = x[:, :, first_scale_feature:first_scale_feature + self.meta.n_scales]
            scales_x[:] = torch.nn.functional.softplus(scales_x)
        elif not self.training and not self.inplace_ops:
            # TODO: CoreMLv4 does not like strided slices.
            # Strides are avoided when switching the first and second dim
            # temporarily.
            x = torch.transpose(x, 1, 2)

            # width
            width_x = x[:, 0:1]

            # classification
            classes_x = x[:, 1:1 + self.meta.n_confidences]
            classes_x = torch.sigmoid(classes_x)

            # regressions x
            first_reg_feature = 1 + self.meta.n_confidences
            regs_x = [
                x[:, first_reg_feature + i * 2:first_reg_feature + (i + 1) * 2]
                for i in range(self.meta.n_vectors)
            ]
            # regressions x: add index
            index_field = index_field_torch(
                (feature_height, feature_width), device=x.device, unsqueeze=(1, 0))
            # TODO: coreml export does not work with the index_field creation in the graph.
            index_field = torch.from_numpy(index_field.numpy())
            regs_x = [reg_x + index_field if do_offset else reg_x
                      for reg_x, do_offset in zip(regs_x, self.meta.vector_offsets)]

            # scale
            first_scale_feature = 1 + self.meta.n_confidences + self.meta.n_vectors * 2
            scales_x = x[:, first_scale_feature:first_scale_feature + self.meta.n_scales]
            scales_x = torch.nn.functional.softplus(scales_x)

            # concat
            x = torch.cat([width_x, classes_x, *regs_x, scales_x], dim=1)

            # TODO: CoreMLv4 problem (see above).
            x = torch.transpose(x, 1, 2)

        return x



@dataclass
class Cif(Base):
    """Head meta data for a Composite Intensity Field (CIF)."""

    keypoints: List[str]
    sigmas: List[float]
    pose: Any = None
    draw_skeleton: List[Tuple[int, int]] = None
    score_weights: List[float] = None

    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 1
    n_scales: ClassVar[int] = 1

    vector_offsets = [True]
    decoder_min_scale = 0.0
    decoder_seed_mask: List[int] = None

    training_weights: List[float] = None

    @property
    def n_fields(self):
        return len(self.keypoints)


@dataclass
class Caf(Base):
    """Head meta data for a Composite Association Field (CAF)."""

    keypoints: List[str]
    sigmas: List[float]
    skeleton: List[Tuple[int, int]]
    pose: Any = None
    sparse_skeleton: List[Tuple[int, int]] = None
    dense_to_sparse_radius: float = 2.0
    only_in_field_of_view: bool = False

    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 2
    n_scales: ClassVar[int] = 2

    vector_offsets = [True, True]
    decoder_min_distance = 0.0
    decoder_max_distance = float('inf')
    decoder_confidence_scales: List[float] = None

    training_weights: List[float] = None

    @property
    def n_fields(self):
        return len(self.skeleton)

    @staticmethod
    def concatenate(metas):
        # TODO: by keypoint name, update skeleton indices if meta.keypoints
        # is not the same for all metas.
        concatenated = Caf(
            name='_'.join(m.name for m in metas),
            dataset=metas[0].dataset,
            keypoints=metas[0].keypoints,
            sigmas=metas[0].sigmas,
            pose=metas[0].pose,
            skeleton=[s for meta in metas for s in meta.skeleton],
            sparse_skeleton=metas[0].sparse_skeleton,
            only_in_field_of_view=metas[0].only_in_field_of_view,
            decoder_confidence_scales=[
                s
                for meta in metas
                for s in (meta.decoder_confidence_scales
                          if meta.decoder_confidence_scales
                          else [1.0 for _ in meta.skeleton])
            ]
        )
        concatenated.head_index = metas[0].head_index
        concatenated.base_stride = metas[0].base_stride
        concatenated.upsample_stride = metas[0].upsample_stride
        return concatenated


@dataclass
class CifDet(Base):
    """Head meta data for a Composite Intensity Field (CIF) for Detection."""

    categories: List[str]

    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 2
    n_scales: ClassVar[int] = 0

    vector_offsets = [True, False]
    decoder_min_scale = 0.0

    training_weights: List[float] = None

    @property
    def n_fields(self):
        return len(self.categories)


@dataclass
class TSingleImageCif(Cif):
    """Single-Image CIF head in tracking models."""


@dataclass
class TSingleImageCaf(Caf):
    """Single-Image CAF head in tracking models."""


@dataclass
class Tcaf(Base):
    """Tracking Composite Association Field."""

    keypoints_single_frame: List[str]
    sigmas_single_frame: List[float]
    pose_single_frame: Any
    draw_skeleton_single_frame: List[Tuple[int, int]] = None
    keypoints: List[str] = None
    sigmas: List[float] = None
    pose: Any = None
    draw_skeleton: List[Tuple[int, int]] = None
    only_in_field_of_view: bool = False

    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 2
    n_scales: ClassVar[int] = 2

    training_weights: List[float] = None

    vector_offsets = [True, True]

    def __post_init__(self):
        if self.keypoints is None:
            self.keypoints = np.concatenate((
                self.keypoints_single_frame,
                self.keypoints_single_frame,
            ), axis=0)
        if self.sigmas is None:
            self.sigmas = np.concatenate((
                self.sigmas_single_frame,
                self.sigmas_single_frame,
            ), axis=0)
        if self.pose is None:
            self.pose = np.concatenate((
                self.pose_single_frame,
                self.pose_single_frame,
            ), axis=0)
        if self.draw_skeleton is None:
            self.draw_skeleton = np.concatenate((
                self.draw_skeleton_single_frame,
                self.draw_skeleton_single_frame,
            ), axis=0)

    @property
    def skeleton(self):
        return [(i + 1, i + 1 + len(self.keypoints_single_frame))
                for i, _ in enumerate(self.keypoints_single_frame)]

    @property
    def n_fields(self):
        return len(self.keypoints_single_frame)