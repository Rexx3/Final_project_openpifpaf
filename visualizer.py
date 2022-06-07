import argparse
from contextlib import contextmanager
import logging
import os
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation
import matplotlib.collections
import matplotlib.patches
import matplotlib.cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import copy
import logging

from heads import Cif, Caf
from annotation import Annotation, AnnotationCrowd
import annotation
class Configurable:
    """Make a class configurable with CLI and by instance.

    .. warning::

        This is an experimental class.
        It is in limited use already but should not be expanded for now.

    To use this class, inherit from it in the class that you want to make
    configurable. There is nothing else to do if your class does not have
    an `__init__` method. If it does, you should take extra keyword arguments
    (`kwargs`) in the signature and pass them to the super constructor.

    Example:

    >>> class MyClass(openpifpaf.Configurable):
    ...     a = 0
    ...     def __init__(self, myclass_argument=None, **kwargs):
    ...         super().__init__(**kwargs)
    ...     def get_a(self):
    ...         return self.a
    >>> MyClass().get_a()
    0

    Instance configurability allows to overwrite a class configuration
    variable with an instance variable by passing that variable as a keyword
    into the class constructor:

    >>> MyClass(a=1).get_a()  # instance variable overwrites value locally
    1
    >>> MyClass().get_a()  # while the class variable is untouched
    0

    """
    def __init__(self, **kwargs):
        # use kwargs to set instance attributes to overwrite class attributes
        for key, value in kwargs.items():
            assert hasattr(self, key), key
            setattr(self, key, value)

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        """Extend an ArgumentParser with the configurable parameters."""

    @classmethod
    def configure(cls, args: argparse.Namespace):
        """Configure the class from parsed command line arguments."""




CMAP_ORANGES_NAN = copy.copy(matplotlib.cm.get_cmap('Oranges'))
CMAP_ORANGES_NAN.set_bad('white', alpha=0.5)

LOG = logging.getLogger(__name__)


def itemsetter(list_, index, value):
    list_[index] = value
    return list_
IMAGENOTGIVEN = object()

class Base:
    all_indices = []
    common_ax = None
    processed_image_intensity_spread = 2.0

    _image = None
    _processed_image = None
    _image_meta = None
    _ground_truth: List[annotation.Base] = None

    def __init__(self, head_name):
        self.head_name = head_name
        self._ax = None

        LOG.debug('%s: indices = %s', head_name, self.indices())

    @classmethod
    def image(cls, image=IMAGENOTGIVEN, meta=None):
        if image is IMAGENOTGIVEN:  # getter
            if callable(Base._image):  # check whether stored value is lazy
                Base._image = Base._image()  # pylint: disable=not-callable
            return Base._image

        if image is None:
            Base._image = None
            Base._image_meta = None
            return cls

        Base._image = lambda: np.asarray(image)
        Base._image_meta = meta
        return cls

    @classmethod
    def processed_image(cls, image=IMAGENOTGIVEN):
        if image is IMAGENOTGIVEN:  # getter
            if callable(Base._processed_image):  # check whether stored value is lazy
                Base._processed_image = Base._processed_image()  # pylint: disable=not-callable
            return Base._processed_image

        if image is None:
            Base._processed_image = None
            return cls

        def process_image(image):
            image = np.moveaxis(np.asarray(image), 0, -1)
            image = np.clip(image / cls.processed_image_intensity_spread * 0.5 + 0.5, 0.0, 1.0)
            return image

        Base._processed_image = lambda: process_image(image)
        return cls

    @staticmethod
    def ground_truth(ground_truth):
        Base._ground_truth = ground_truth

    @staticmethod
    def reset():
        Base._image = None
        Base._image_meta = None
        Base._processed_image = None
        Base._ground_truth = None

    @classmethod
    def normalized_index(cls, data):
        if isinstance(data, str):
            data = data.split(':')

        # unpack comma separation
        for di, d in enumerate(data):
            if ',' not in d:
                continue
            multiple = [cls.normalized_index(itemsetter(data, di, unpacked))
                        for unpacked in d.split(',')]
            # flatten before return
            return [item for items in multiple for item in items]

        if len(data) >= 2 and len(data[1]) == 0:
            data[1] = -1

        if len(data) == 3:
            return [(data[0], int(data[1]), data[2])]
        if len(data) == 2:
            return [(data[0], int(data[1]), 'all')]
        return [(data[0], -1, 'all')]

    @classmethod
    def set_all_indices(cls, all_indices):
        cls.all_indices = [d for dd in all_indices for d in cls.normalized_index(dd)]

    def indices(self, type_=None, with_all=True):
        head_names = self.head_name
        if not isinstance(head_names, (tuple, list)):
            head_names = (head_names,)
        return [
            f for hn, f, r_type in self.all_indices
            if hn in head_names and (
                type_ is None
                or (with_all and r_type == 'all')
                or type_ == r_type
            )
        ]

    @staticmethod
    def colorbar(ax, colored_element, size='3%', pad=0.01):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size=size, pad=pad)
        cb = plt.colorbar(colored_element, cax=cax)
        cb.outline.set_linewidth(0.1)

    @contextmanager
    def image_canvas(self, image, *args, **kwargs):
        ax = self._ax or self.common_ax
        if ax is not None:
            ax.set_axis_off()
            ax.imshow(np.asarray(image))
            yield ax
            return

        with image_canvas(image, *args, **kwargs) as ax:
            yield ax

    @contextmanager
    def canvas(self, *args, **kwargs):
        ax = self._ax or self.common_ax
        if ax is not None:
            yield ax
            return

        with canvas(*args, **kwargs) as ax:
            yield ax

    @staticmethod
    def scale_scalar(field, stride):
        field = np.repeat(field, stride, 0)
        field = np.repeat(field, stride, 1)

        # center (the result is technically still off by half a pixel)
        half_stride = stride // 2
        return field[half_stride:-half_stride + 1, half_stride:-half_stride + 1]

class CifVisualizer(Base):
    """Visualize a CIF field."""

    def __init__(self, meta: Cif):
        super().__init__(meta.name)
        self.meta = meta
        keypoint_painter = KeypointPainter(monocolor_connections=True)
        self.annotation_painter = AnnotationPainter(painters={'Annotation': keypoint_painter})

    def targets(self, field, *, annotation_dicts):
        assert self.meta.keypoints is not None
        assert self.meta.draw_skeleton is not None

        annotations = [
            Annotation(
                keypoints=self.meta.keypoints,
                skeleton=self.meta.draw_skeleton,
                sigmas=self.meta.sigmas,
                score_weights=self.meta.score_weights
            ).set(
                ann['keypoints'], fixed_score='', fixed_bbox=ann['bbox'])
            if not ann['iscrowd']
            else AnnotationCrowd(['keypoints']).set(1, ann['bbox'])
            for ann in annotation_dicts
        ]

        self._confidences(field[:, 0])
        self._regressions(field[:, 1:3], field[:, 4], annotations=annotations)

    def predicted(self, field):
        self._confidences(field[:, 1])
        self._regressions(field[:, 2:4], field[:, 4],
                          annotations=self._ground_truth,
                          confidence_fields=field[:, 1],
                          uv_is_offset=False)

    def _confidences(self, confidences):
        for f in self.indices('confidence'):
            LOG.debug('%s', self.meta.keypoints[f])

            with self.image_canvas(self.processed_image(), margin=[0.0, 0.01, 0.05, 0.01]) as ax:
                im = ax.imshow(self.scale_scalar(confidences[f], self.meta.stride),
                               alpha=0.9, vmin=0.0, vmax=1.0, cmap=CMAP_ORANGES_NAN)
                self.colorbar(ax, im)

    def _regressions(self, regression_fields, scale_fields, *,
                     annotations=None, confidence_fields=None, uv_is_offset=True):
        for f in self.indices('regression'):
            LOG.debug('%s', self.meta.keypoints[f])
            confidence_field = confidence_fields[f] if confidence_fields is not None else None

            with self.image_canvas(self.processed_image(), margin=[0.0, 0.01, 0.05, 0.01]) as ax:
                white_screen(ax, alpha=0.5)
                if annotations:
                    self.annotation_painter.annotations(ax, annotations, color='lightgray')
                q = quiver(ax,
                                regression_fields[f, :2],
                                confidence_field=confidence_field,
                                xy_scale=self.meta.stride, uv_is_offset=uv_is_offset,
                                cmap='Oranges', clim=(0.5, 1.0), width=0.001)
                boxes(ax, scale_fields[f] / 2.0,
                           confidence_field=confidence_field,
                           regression_field=regression_fields[f, :2],
                           xy_scale=self.meta.stride, cmap='Oranges', fill=False,
                           regression_field_is_offset=uv_is_offset)
                if f in self.indices('margin', with_all=False):
                    margins(ax, regression_fields[f, :6], xy_scale=self.meta.stride)

                self.colorbar(ax, q)


class CafVisualizer(Base):
    """Visualize CAF field."""

    def __init__(self, meta: Caf):
        super().__init__(meta.name)
        self.meta = meta
        keypoint_painter = KeypointPainter(monocolor_connections=True)
        self.annotation_painter = AnnotationPainter(painters={'Annotation': keypoint_painter})

    def targets(self, field, *, annotation_dicts):
        assert self.meta.keypoints is not None
        assert self.meta.skeleton is not None

        annotations = [
            Annotation(
                keypoints=self.meta.keypoints,
                skeleton=self.meta.skeleton,
                sigmas=self.meta.sigmas,
            ).set(
                ann['keypoints'], fixed_score='', fixed_bbox=ann['bbox'])
            if not ann['iscrowd']
            else AnnotationCrowd(['keypoints']).set(1, ann['bbox'])
            for ann in annotation_dicts
        ]

        self._confidences(field[:, 0])
        self._regressions(field[:, 1:3], field[:, 3:5], field[:, 7], field[:, 8],
                          annotations=annotations)

    def predicted(self, field):
        self._confidences(field[:, 1])
        self._regressions(field[:, 2:4], field[:, 4:6], field[:, 6], field[:, 7],
                          annotations=self._ground_truth,
                          confidence_fields=field[:, 1],
                          uv_is_offset=False)

    def _confidences(self, confidences):
        for f in self.indices('confidence'):
            LOG.debug('%s,%s',
                      self.meta.keypoints[self.meta.skeleton[f][0] - 1],
                      self.meta.keypoints[self.meta.skeleton[f][1] - 1])

            with self.image_canvas(self.processed_image(), margin=[0.0, 0.01, 0.05, 0.01]) as ax:
                im = ax.imshow(self.scale_scalar(confidences[f], self.meta.stride),
                               alpha=0.9, vmin=0.0, vmax=1.0, cmap=CMAP_BLUES_NAN)
                self.colorbar(ax, im)

    def _regressions(self, regression_fields1, regression_fields2,
                     scale_fields1, scale_fields2, *,
                     annotations=None, confidence_fields=None, uv_is_offset=True):
        for f in self.indices('regression'):
            LOG.debug('%s,%s',
                      self.meta.keypoints[self.meta.skeleton[f][0] - 1],
                      self.meta.keypoints[self.meta.skeleton[f][1] - 1])
            confidence_field = confidence_fields[f] if confidence_fields is not None else None

            with self.image_canvas(self.processed_image(), margin=[0.0, 0.01, 0.05, 0.01]) as ax:
                white_screen(ax, alpha=0.5)
                if annotations:
                    self.annotation_painter.annotations(ax, annotations, color='lightgray')
                q1 = quiver(ax,
                                 regression_fields1[f, :2],
                                 confidence_field=confidence_field,
                                 xy_scale=self.meta.stride, uv_is_offset=uv_is_offset,
                                 cmap='Blues', clim=(0.5, 1.0), width=0.001)
                quiver(ax,
                            regression_fields2[f, :2],
                            confidence_field=confidence_field,
                            xy_scale=self.meta.stride, uv_is_offset=uv_is_offset,
                            cmap='Greens', clim=(0.5, 1.0), width=0.001)
                boxes(ax, scale_fields1[f] / 2.0,
                           confidence_field=confidence_field,
                           regression_field=regression_fields1[f, :2],
                           xy_scale=self.meta.stride, cmap='Blues', fill=False,
                           regression_field_is_offset=uv_is_offset)
                boxes(ax, scale_fields2[f] / 2.0,
                           confidence_field=confidence_field,
                           regression_field=regression_fields2[f, :2],
                           xy_scale=self.meta.stride, cmap='Greens', fill=False,
                           regression_field_is_offset=uv_is_offset)

                self.colorbar(ax, q1)


class DetectionPainter:
    def __init__(self, *, xy_scale=1.0):
        self.xy_scale = xy_scale

    def annotation(self, ax, ann, *, color=None, text=None, subtext=None):
        if color is None:
            color = 0
        if isinstance(color, (int, np.integer)):
            color = matplotlib.cm.get_cmap('tab20')((color % 20 + 0.05) / 20)

        if text is None:
            text = ann.category
            if getattr(ann, 'id_', None):
                text += ' ({})'.format(ann.id_)
        if subtext is None and ann.score:
            subtext = '{:.0%}'.format(ann.score)

        x, y, w, h = ann.bbox * self.xy_scale
        if w < 5.0:
            x -= 2.0
            w += 4.0
        if h < 5.0:
            y -= 2.0
            h += 4.0

        # draw box
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (x, y), w, h, fill=False, color=color, linewidth=1.0))

        # draw text
        ax.annotate(
            text,
            (x, y),
            fontsize=8,
            xytext=(5.0, 5.0),
            textcoords='offset points',
            color='white', bbox={'facecolor': color, 'alpha': 0.5, 'linewidth': 0},
        )
        if subtext is not None:
            ax.annotate(
                subtext,
                (x, y),
                fontsize=5,
                xytext=(5.0, 18.0 + 3.0),
                textcoords='offset points',
                color='white', bbox={'facecolor': color, 'alpha': 0.5, 'linewidth': 0},
            )


class CrowdPainter:
    def __init__(self, *, xy_scale=1.0):
        self.xy_scale = xy_scale

    @staticmethod
    def draw_polygon(ax, outlines, *, alpha=0.5, color='orange'):
        for outline in outlines:
            assert outline.shape[1] == 2

        patches = []
        for outline in outlines:
            polygon = matplotlib.patches.Polygon(
                outline[:, :2], color=color, facecolor=color, alpha=alpha)
            patches.append(polygon)
        ax.add_collection(matplotlib.collections.PatchCollection(patches, match_original=True))

    def annotation(self, ax, ann, *, color=None, text=None, subtext=None):
        if color is None:
            color = 0
        if isinstance(color, (int, np.integer)):
            color = matplotlib.cm.get_cmap('tab20')((color % 20 + 0.05) / 20)

        if text is None:
            text = '{} (crowd)'.format(ann.category)
            if getattr(ann, 'id_', None):
                text += ' ({})'.format(ann.id_)

        x, y, w, h = ann.bbox * self.xy_scale
        if w < 5.0:
            x -= 2.0
            w += 4.0
        if h < 5.0:
            y -= 2.0
            h += 4.0

        # draw box
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (x, y), w, h, fill=False, color=color, linewidth=1.0, linestyle='dotted'))

        # draw text
        ax.annotate(
            text,
            (x, y),
            fontsize=8,
            xytext=(5.0, 5.0),
            textcoords='offset points',
            color='white', bbox={'facecolor': color, 'alpha': 0.5, 'linewidth': 0},
        )
        if subtext is not None:
            ax.annotate(
                subtext,
                (x, y),
                fontsize=5,
                xytext=(5.0, 18.0 + 3.0),
                textcoords='offset points',
                color='white', bbox={'facecolor': color, 'alpha': 0.5, 'linewidth': 0},
            )


class KeypointPainter(Configurable):
    """Paint poses.

    The constructor can take any class attribute as parameter and
    overwrite the global default for that instance.

    Example to create a KeypointPainter with thick lines:

    >>> kp = KeypointPainter(line_width=48)

    """

    show_box = False
    show_joint_confidences = False
    show_joint_scales = False
    show_decoding_order = False
    show_frontier_order = False
    show_only_decoded_connections = False

    textbox_alpha = 0.5
    text_color = 'white'
    monocolor_connections = False
    line_width = None
    marker_size = None
    solid_threshold = 0.5
    font_size = 8

    def __init__(self, *,
                 xy_scale=1.0,
                 highlight=None,
                 highlight_invisible=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.xy_scale = xy_scale
        self.highlight = highlight
        self.highlight_invisible = highlight_invisible

        # set defaults for line_width and marker_size depending on monocolor
        if self.line_width is None:
            self.line_width = 2 if self.monocolor_connections else 6
        if self.marker_size is None:
            if self.monocolor_connections:
                self.marker_size = max(self.line_width + 1, int(self.line_width * 3.0))
            else:
                self.marker_size = max(1, int(self.line_width * 0.5))

        LOG.debug('color connections = %s, lw = %d, marker = %d',
                  self.monocolor_connections, self.line_width, self.marker_size)

    def _draw_skeleton(self, ax, x, y, v, *,
                       skeleton, skeleton_mask=None, color=None, alpha=1.0, **kwargs):
        if not np.any(v > 0):
            return

        if skeleton_mask is None:
            skeleton_mask = [True for _ in skeleton]
        assert len(skeleton) == len(skeleton_mask)

        # connections
        lines, line_colors, line_styles = [], [], []
        for ci, ((j1i, j2i), mask) in enumerate(zip(np.array(skeleton) - 1, skeleton_mask)):
            if not mask:
                continue
            c = color
            if not self.monocolor_connections:
                c = matplotlib.cm.get_cmap('tab20')((ci % 20 + 0.05) / 20)
            if v[j1i] > 0 and v[j2i] > 0:
                lines.append([(x[j1i], y[j1i]), (x[j2i], y[j2i])])
                line_colors.append(c)
                if v[j1i] > self.solid_threshold and v[j2i] > self.solid_threshold:
                    line_styles.append('solid')
                else:
                    line_styles.append('dashed')
        ax.add_collection(matplotlib.collections.LineCollection(
            lines, colors=line_colors,
            linewidths=kwargs.get('linewidth', self.line_width),
            linestyles=kwargs.get('linestyle', line_styles),
            capstyle='round',
            alpha=alpha,
        ))

        # joints
        ax.scatter(
            x[v > 0.0], y[v > 0.0], s=self.marker_size**2, marker='.',
            color=color if self.monocolor_connections else 'white',
            edgecolor='k' if self.highlight_invisible else None,
            zorder=2,
            alpha=alpha,
        )

        # highlight joints
        if self.highlight is not None:
            highlight_v = np.zeros_like(v)
            highlight_v[self.highlight] = 1
            highlight_v = np.logical_and(v, highlight_v)

            ax.scatter(
                x[highlight_v], y[highlight_v], s=self.marker_size**2, marker='.',
                color=color if self.monocolor_connections else 'white',
                edgecolor='k' if self.highlight_invisible else None,
                zorder=2,
                alpha=alpha,
            )

    def keypoints(self, ax, keypoint_sets, *,
                  skeleton, scores=None, color=None, colors=None, texts=None):
        if keypoint_sets is None:
            return

        if color is None and colors is None:
            colors = range(len(keypoint_sets))

        for i, kps in enumerate(np.asarray(keypoint_sets)):
            assert kps.shape[1] == 3
            x = kps[:, 0] * self.xy_scale
            y = kps[:, 1] * self.xy_scale
            v = kps[:, 2]

            if colors is not None:
                color = colors[i]

            if isinstance(color, (int, np.integer)):
                color = matplotlib.cm.get_cmap('tab20')((color % 20 + 0.05) / 20)

            self._draw_skeleton(ax, x, y, v, skeleton=skeleton, color=color)
            if self.show_box:
                score = scores[i] if scores is not None else None
                self._draw_box(ax, x, y, v, color, score)

            if texts is not None:
                self._draw_text(ax, x, y, v, texts[i], color)

    @staticmethod
    def _draw_box(ax, x, y, w, h, color, score=None, linewidth=1):
        if w < 5.0:
            x -= 2.0
            w += 4.0
        if h < 5.0:
            y -= 2.0
            h += 4.0
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (x, y), w, h, fill=False, color=color, linewidth=linewidth))

        if score:
            ax.text(x, y - linewidth, '{:.4f}'.format(score), fontsize=8, color=color)

    @classmethod
    def _draw_text(cls, ax, x, y, v, text, color, *, subtext=None, alpha=1.0):
        if cls.font_size == 0:
            return
        if not np.any(v > 0):
            return

        coord_i = np.argsort(y[v > 0])
        if np.sum(v > 0) >= 2 \
           and y[v > 0][coord_i[1]] < y[v > 0][coord_i[0]] + 10:
            # second coordinate within 10 pixels
            f0 = 0.5 + 0.5 * (y[v > 0][coord_i[1]] - y[v > 0][coord_i[0]]) / 10.0
            coord_y = f0 * y[v > 0][coord_i[0]] + (1.0 - f0) * y[v > 0][coord_i[1]]
            coord_x = f0 * x[v > 0][coord_i[0]] + (1.0 - f0) * x[v > 0][coord_i[1]]
        else:
            coord_y = y[v > 0][coord_i[0]]
            coord_x = x[v > 0][coord_i[0]]

        bbox_config = {'facecolor': color, 'alpha': alpha * cls.textbox_alpha, 'linewidth': 0}
        ax.annotate(
            text,
            (coord_x, coord_y),
            fontsize=cls.font_size,
            xytext=(5.0, 5.0),
            textcoords='offset points',
            color=cls.text_color,
            bbox=bbox_config,
            alpha=alpha,
        )
        if subtext is not None:
            ax.annotate(
                subtext,
                (coord_x, coord_y),
                fontsize=cls.font_size * 5 // 8,
                xytext=(5.0, 18.0 + 3.0),
                textcoords='offset points',
                color=cls.text_color,
                bbox=bbox_config,
                alpha=alpha,
            )

    @staticmethod
    def _draw_scales(ax, xs, ys, vs, color, scales, alpha=1.0):
        for x, y, v, scale in zip(xs, ys, vs, scales):
            if v == 0.0:
                continue
            ax.add_patch(
                matplotlib.patches.Rectangle(
                    (x - scale / 2, y - scale / 2), scale, scale,
                    fill=False, color=color, alpha=alpha))

    @classmethod
    def _draw_joint_confidences(cls, ax, xs, ys, vs, color):
        for x, y, v in zip(xs, ys, vs):
            if v == 0.0:
                continue
            ax.annotate(
                '{:.0%}'.format(v),
                (x, y),
                fontsize=6,
                xytext=(0.0, 0.0),
                textcoords='offset points',
                verticalalignment='top',
                color=cls.text_color,
                bbox={'facecolor': color, 'alpha': 0.2, 'linewidth': 0, 'pad': 0.0},
            )

    def annotation(self, ax, ann, *, color=None, text=None, subtext=None, alpha=1.0):
        if color is None:
            color = 0
        if isinstance(color, (int, np.integer)):
            color = matplotlib.cm.get_cmap('tab20')((color % 20 + 0.05) / 20)

        text_is_score = False
        if text is None and hasattr(ann, 'id_'):
            text = '{}'.format(ann.id_)
        if text is None and getattr(ann, 'score', None):
            text = '{:.0%}'.format(ann.score)
            text_is_score = True
        if subtext is None and not text_is_score and getattr(ann, 'score', None):
            subtext = '{:.0%}'.format(ann.score)

        kps = ann.data
        assert kps.shape[1] == 3
        x = kps[:, 0] * self.xy_scale
        y = kps[:, 1] * self.xy_scale
        v = kps[:, 2]

        if self.show_frontier_order:
            frontier = set((s, e) for s, e in ann.frontier_order)
            frontier_skeleton_mask = [
                (s - 1, e - 1) in frontier or (e - 1, s - 1) in frontier
                for s, e in ann.skeleton
            ]
            frontier_skeleton = [se for se, m in zip(ann.skeleton, frontier_skeleton_mask) if m]
            self._draw_skeleton(ax, x, y, v, color='black', skeleton=frontier_skeleton,
                                linestyle='dotted', linewidth=1)

        skeleton_mask = None
        if self.show_only_decoded_connections:
            decoded_connections = set((jsi, jti) for jsi, jti, _, __ in ann.decoding_order)
            skeleton_mask = [
                (s - 1, e - 1) in decoded_connections or (e - 1, s - 1) in decoded_connections
                for s, e in ann.skeleton
            ]

        self._draw_skeleton(ax, x, y, v, color=color,
                            skeleton=ann.skeleton, skeleton_mask=skeleton_mask, alpha=alpha)

        if self.show_joint_scales and ann.joint_scales is not None:
            self._draw_scales(ax, x, y, v, color, ann.joint_scales, alpha=alpha)

        if self.show_joint_confidences:
            self._draw_joint_confidences(ax, x, y, v, color)

        if self.show_box:
            x_, y_, w_, h_ = [v * self.xy_scale for v in ann.bbox()]
            self._draw_box(ax, x_, y_, w_, h_, color, ann.score)

        if text is not None:
            self._draw_text(ax, x, y, v, text, color, subtext=subtext, alpha=alpha)

        if self.show_decoding_order and hasattr(ann, 'decoding_order'):
            self._draw_decoding_order(ax, ann.decoding_order)

    @staticmethod
    def _draw_decoding_order(ax, decoding_order):
        for step_i, (jsi, jti, jsxyv, jtxyv) in enumerate(decoding_order):
            ax.plot([jsxyv[0], jtxyv[0]], [jsxyv[1], jtxyv[1]], '--', color='black')
            ax.text(0.5 * (jsxyv[0] + jtxyv[0]), 0.5 * (jsxyv[1] + jtxyv[1]),
                    '{}: {} -> {}'.format(step_i, jsi, jti), fontsize=8,
                    color='white', bbox={'facecolor': 'black', 'alpha': 0.5, 'linewidth': 0})

PAINTERS = {
    'Annotation': KeypointPainter,
    'AnnotationCrowd': CrowdPainter,
    'AnnotationDet': DetectionPainter,
}

class AnnotationPainter:
    def __init__(self, *,
                 xy_scale=1.0,
                 painters=None):
        self.painters = {annotation_type: painter(xy_scale=xy_scale)
                         for annotation_type, painter in PAINTERS.items()}

        if painters:
            for annotation_type, painter in painters.items():
                self.painters[annotation_type] = painter

    def annotations(self, ax, annotations, *,
                    color=None, colors=None, texts=None, subtexts=None, **kwargs):
        for i, ann in enumerate(annotations):
            if colors is not None:
                this_color = colors[i]
            elif color is not None:
                this_color = color
            elif getattr(ann, 'id_', None):
                this_color = ann.id_
            else:
                this_color = i

            text = None
            if texts is not None:
                text = texts[i]

            subtext = None
            if subtexts is not None:
                subtext = subtexts[i]

            painter = self.painters[ann.__class__.__name__]
            painter.annotation(ax, ann, color=this_color, text=text, subtext=subtext, **kwargs)


class Canvas:
    """Canvas for plotting.

    All methods expose Axes objects. To get Figure objects, you can ask the axis
    `ax.get_figure()`.
    """

    all_images_directory = None
    all_images_count = 0
    show = False
    image_width = 7.0
    image_height = None
    blank_dpi = 200
    image_dpi_factor = 2.0
    image_min_dpi = 50.0
    out_file_extension = 'jpeg'
    white_overlay = False

    @classmethod
    def generic_name(cls):
        if cls.all_images_directory is None:
            return None

        os.makedirs(cls.all_images_directory, exist_ok=True)

        cls.all_images_count += 1
        return os.path.join(cls.all_images_directory,
                            '{:04}.{}'.format(cls.all_images_count, cls.out_file_extension))

    @classmethod
    @contextmanager
    def blank(cls, fig_file=None, *, dpi=None, nomargin=False, **kwargs):
        if plt is None:
            raise Exception('please install matplotlib')
        if fig_file is None:
            fig_file = cls.generic_name()
        if dpi is None:
            dpi = cls.blank_dpi

        if 'figsize' not in kwargs:
            kwargs['figsize'] = (10, 6)

        if nomargin:
            if 'gridspec_kw' not in kwargs:
                kwargs['gridspec_kw'] = {}
            kwargs['gridspec_kw']['wspace'] = 0
            kwargs['gridspec_kw']['hspace'] = 0
            kwargs['gridspec_kw']['left'] = 0.0
            kwargs['gridspec_kw']['right'] = 1.0
            kwargs['gridspec_kw']['top'] = 1.0
            kwargs['gridspec_kw']['bottom'] = 0.0
        fig, ax = plt.subplots(dpi=dpi, **kwargs)

        yield ax

        fig.set_tight_layout(not nomargin)
        if fig_file:
            LOG.debug('writing image to %s', fig_file)
            fig.savefig(fig_file)
        if cls.show:
            plt.show()
        plt.close(fig)

    @classmethod
    @contextmanager
    def image(cls, image, fig_file=None, *, margin=None, **kwargs):
        if plt is None:
            raise Exception('please install matplotlib')
        if fig_file is None:
            fig_file = cls.generic_name()

        image = np.asarray(image)

        if margin is None:
            margin = [0.0, 0.0, 0.0, 0.0]
        elif isinstance(margin, float):
            margin = [margin, margin, margin, margin]
        assert len(margin) == 4

        if 'figsize' not in kwargs:
            # compute figure size: use image ratio and take the drawable area
            # into account that is left after subtracting margins.
            image_ratio = image.shape[0] / image.shape[1]
            image_area_ratio = (1.0 - margin[1] - margin[3]) / (1.0 - margin[0] - margin[2])
            if cls.image_width is not None:
                kwargs['figsize'] = (
                    cls.image_width,
                    cls.image_width * image_ratio / image_area_ratio
                )
            elif cls.image_height:
                kwargs['figsize'] = (
                    cls.image_height * image_area_ratio / image_ratio,
                    cls.image_height
                )

        dpi = max(cls.image_min_dpi, image.shape[1] / kwargs['figsize'][0] * cls.image_dpi_factor)
        fig = plt.figure(dpi=dpi, **kwargs)
        ax = plt.Axes(fig, [0.0 + margin[0],
                            0.0 + margin[1],
                            1.0 - margin[2],
                            1.0 - margin[3]])
        ax.set_axis_off()
        ax.set_xlim(-0.5, image.shape[1] - 0.5)  # imshow uses center-pixel-coordinates
        ax.set_ylim(image.shape[0] - 0.5, -0.5)
        fig.add_axes(ax)
        ax.imshow(image)
        if cls.white_overlay:
            white_screen(ax, cls.white_overlay)

        yield ax

        if fig_file:
            LOG.debug('writing image to %s', fig_file)
            fig.savefig(fig_file)
        if cls.show:
            plt.show()
        plt.close(fig)

    @classmethod
    @contextmanager
    def annotation(cls, ann, *,
                   filename=None,
                   margin=0.5,
                   fig_w=None,
                   fig_h=5.0,
                   **kwargs):
        bbox = ann.bbox()
        xlim = bbox[0] - margin, bbox[0] + bbox[2] + margin
        ylim = bbox[1] - margin, bbox[1] + bbox[3] + margin
        if fig_w is None:
            fig_w = fig_h / (ylim[1] - ylim[0]) * (xlim[1] - xlim[0])
        fig_w *= kwargs.get('ncols', 1)
        fig_h *= kwargs.get('nrows', 1)

        with cls.blank(filename, figsize=(fig_w, fig_h), nomargin=True, **kwargs) as ax:
            iter_ax = [ax] if hasattr(ax, 'set_axis_off') else ax
            for ax_ in iter_ax:
                ax_.set_axis_off()
                ax_.set_xlim(*xlim)
                ax_.set_ylim(*ylim)

            yield ax


# keep previous interface for now:
canvas = Canvas.blank
image_canvas = Canvas.image
annotation_canvas = Canvas.annotation


def white_screen(ax, alpha=0.9):
    ax.add_patch(
        plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, alpha=alpha,
                      facecolor='white')
    )

def quiver(ax, vector_field, *,
           confidence_field=None, step=1, threshold=0.5,
           xy_scale=1.0, uv_is_offset=False,
           reg_uncertainty=None, **kwargs):
    x, y, u, v, c, r = [], [], [], [], [], []
    for j in range(0, vector_field.shape[1], step):
        for i in range(0, vector_field.shape[2], step):
            if confidence_field is not None and confidence_field[j, i] < threshold:
                continue
            x.append(i * xy_scale)
            y.append(j * xy_scale)
            u.append(vector_field[0, j, i] * xy_scale)
            v.append(vector_field[1, j, i] * xy_scale)
            c.append(confidence_field[j, i] if confidence_field is not None else 1.0)
            r.append(reg_uncertainty[j, i] * xy_scale if reg_uncertainty is not None else None)
    x = np.array(x)
    y = np.array(y)
    u = np.array(u)
    v = np.array(v)
    c = np.array(c)
    r = np.array(r)
    s = np.argsort(c)
    if uv_is_offset:
        u += x
        v += y

    for uu, vv, rr in zip(u, v, r):
        if not rr:
            continue
        circle = matplotlib.patches.Circle(
            (uu, vv), rr / 2.0, zorder=11, linewidth=1, alpha=1.0,
            fill=False, color='orange')
        ax.add_artist(circle)

    return ax.quiver(x[s], y[s], u[s] - x[s], v[s] - y[s], c[s],
                     angles='xy', scale_units='xy', scale=1, zorder=10, **kwargs)


def margins(ax, vector_field, *,
            confidence_field=None, step=1, threshold=0.5,
            xy_scale=1.0, uv_is_offset=False, **kwargs):
    x, y, u, v, r = [], [], [], [], []
    for j in range(0, vector_field.shape[1], step):
        for i in range(0, vector_field.shape[2], step):
            if confidence_field is not None and confidence_field[j, i] < threshold:
                continue
            x.append(i * xy_scale)
            y.append(j * xy_scale)
            u.append(vector_field[0, j, i] * xy_scale)
            v.append(vector_field[1, j, i] * xy_scale)
            r.append(vector_field[2:6, j, i] * xy_scale)
    x = np.array(x)
    y = np.array(y)
    u = np.array(u)
    v = np.array(v)
    r = np.array(r)
    if uv_is_offset:
        u -= x
        v -= y

    wedge_angles = [
        (0.0, 90.0),
        (90.0, 180.0),
        (270.0, 360.0),
        (180.0, 270.0),
    ]

    for xx, yy, uu, vv, rr in zip(x, y, u, v, r):
        for q_rr, (theta1, theta2) in zip(rr, wedge_angles):
            if not np.isfinite(q_rr):
                continue
            wedge = matplotlib.patches.Wedge(
                (xx + uu, yy + vv), q_rr, theta1, theta2,
                zorder=9, linewidth=1, alpha=0.5 / 16.0,
                fill=True, color='orange', **kwargs)
            ax.add_artist(wedge)


def arrows(ax, fourd, xy_scale=1.0, threshold=0.0, **kwargs):
    mask = np.min(fourd[:, 2], axis=0) >= threshold
    fourd = fourd[:, :, mask]
    (x1, y1), (x2, y2) = fourd[:, :2, :] * xy_scale
    c = np.min(fourd[:, 2], axis=0)
    s = np.argsort(c)
    return ax.quiver(x1[s], y1[s], (x2 - x1)[s], (y2 - y1)[s], c[s],
                     angles='xy', scale_units='xy', scale=1, zorder=10, **kwargs)


def boxes(ax, sigma_field, **kwargs):
    boxes_wh(ax, sigma_field * 2.0, sigma_field * 2.0, **kwargs)


def boxes_wh(ax, w_field, h_field, *, confidence_field=None, regression_field=None,
             xy_scale=1.0, step=1, threshold=0.5,
             regression_field_is_offset=False,
             cmap='viridis_r', clim=(0.5, 1.0), linewidth=1, **kwargs):
    x, y, w, h, c = [], [], [], [], []
    for j in range(0, w_field.shape[0], step):
        for i in range(0, w_field.shape[1], step):
            if confidence_field is not None and confidence_field[j, i] < threshold:
                continue
            x_offset, y_offset = 0.0, 0.0
            if regression_field is not None:
                x_offset = regression_field[0, j, i]
                y_offset = regression_field[1, j, i]
                if not regression_field_is_offset:
                    x_offset = x_offset - i
                    y_offset = y_offset - j
            x.append((i + x_offset) * xy_scale)
            y.append((j + y_offset) * xy_scale)
            w.append(w_field[j, i] * xy_scale)
            h.append(h_field[j, i] * xy_scale)
            c.append(confidence_field[j, i] if confidence_field is not None else 1.0)

    cmap = matplotlib.cm.get_cmap(cmap)
    cnorm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
    for xx, yy, ww, hh, cc in zip(x, y, w, h, c):
        color = cmap(cnorm(cc))
        rectangle = matplotlib.patches.Rectangle(
            (xx - ww / 2.0, yy - hh / 2.0), ww, hh,
            color=color, zorder=10, linewidth=linewidth, **kwargs)
        ax.add_artist(rectangle)


def circles(ax, radius_field, *, confidence_field=None, regression_field=None,
            xy_scale=1.0, step=1, threshold=0.5,
            regression_field_is_offset=False,
            cmap='viridis_r', clim=(0.5, 1.0), **kwargs):
    x, y, s, c = [], [], [], []
    for j in range(0, radius_field.shape[0], step):
        for i in range(0, radius_field.shape[1], step):
            if confidence_field is not None and confidence_field[j, i] < threshold:
                continue
            x_offset, y_offset = 0.0, 0.0
            if regression_field is not None:
                x_offset = regression_field[0, j, i]
                y_offset = regression_field[1, j, i]
                if not regression_field_is_offset:
                    x_offset -= i
                    y_offset -= j
            x.append((i + x_offset) * xy_scale)
            y.append((j + y_offset) * xy_scale)
            s.append(radius_field[j, i] * xy_scale)
            c.append(confidence_field[j, i] if confidence_field is not None else 1.0)

    cmap = matplotlib.cm.get_cmap(cmap)
    cnorm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
    for xx, yy, ss, cc in zip(x, y, s, c):
        color = cmap(cnorm(cc))
        circle = matplotlib.patches.Circle(
            (xx, yy), ss,
            color=color, zorder=10, linewidth=1, **kwargs)
        ax.add_artist(circle)
