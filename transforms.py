from abc import ABCMeta, abstractmethod
from typing import List
import copy
import logging

import numpy as np
import torch
import cv2
import PIL
import torchvision
import encoder
import warnings
import scipy
import scipy.ndimage
import itertools
import math
import io
LOG = logging.getLogger(__name__)


def rotate_box(bbox, width, height, angle_degrees):
    """Input bounding box is of the form x, y, width, height."""

    cangle = math.cos(angle_degrees / 180.0 * math.pi)
    sangle = math.sin(angle_degrees / 180.0 * math.pi)

    four_corners = np.array([
        [bbox[0], bbox[1]],
        [bbox[0] + bbox[2], bbox[1]],
        [bbox[0], bbox[1] + bbox[3]],
        [bbox[0] + bbox[2], bbox[1] + bbox[3]],
    ])

    x_old = four_corners[:, 0].copy() - width / 2
    y_old = four_corners[:, 1].copy() - height / 2
    four_corners[:, 0] = width / 2 + cangle * x_old + sangle * y_old
    four_corners[:, 1] = height / 2 - sangle * x_old + cangle * y_old

    x = np.min(four_corners[:, 0])
    y = np.min(four_corners[:, 1])
    xmax = np.max(four_corners[:, 0])
    ymax = np.max(four_corners[:, 1])

    return np.array([x, y, xmax - x, ymax - y])

class Preprocess(metaclass=ABCMeta):
    """Preprocess an image with annotations and meta information."""
    @abstractmethod
    def __call__(self, image, anns, meta):
        """Implementation of preprocess operation."""

class Compose(Preprocess):
    """Execute given transforms in sequential order."""
    def __init__(self, preprocess_list: List[Preprocess]):
        self.preprocess_list = preprocess_list

    def __call__(self, *args):
        for p in self.preprocess_list:
            if p is None:
                continue
            args = p(*args)

        return args




class NormalizeAnnotations(Preprocess):
    @classmethod
    def normalize_annotations(cls, anns):
        anns = copy.deepcopy(anns)

        for ann in anns:
            # if isinstance(ann, annotation.Base):
            #     # already converted to an annotation type
            #     continue

            if 'keypoints' not in ann:
                ann['keypoints'] = []
            if 'iscrowd' not in ann:
                ann['iscrowd'] = False

            ann['keypoints'] = np.asarray(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
            if 'bbox' not in ann:
                ann['bbox'] = cls.bbox_from_keypoints(ann['keypoints'])
            ann['bbox'] = np.asarray(ann['bbox'], dtype=np.float32)
            if 'bbox_original' not in ann:
                ann['bbox_original'] = np.copy(ann['bbox'])
            if 'segmentation' in ann:
                del ann['segmentation']

        return anns

    @staticmethod
    def bbox_from_keypoints(keypoints):
        visible_keypoints = keypoints[keypoints[:, 2] > 0.0]
        if not visible_keypoints.shape[0]:
            return [0, 0, 0, 0]

        x1 = np.min(visible_keypoints[:, 0])
        y1 = np.min(visible_keypoints[:, 1])
        x2 = np.max(visible_keypoints[:, 0])
        y2 = np.max(visible_keypoints[:, 1])
        return [x1, y1, x2 - x2, y2 - y1]

    def __call__(self, image, anns, meta):
        anns = self.normalize_annotations(anns)

        if meta is None:
            meta = {}

        # fill meta with defaults if not already present
        w, h = image.size
        meta_from_image = {
            'offset': np.array((0.0, 0.0)),
            'scale': np.array((1.0, 1.0)),
            'rotation': {'angle': 0.0, 'width': None, 'height': None},
            'valid_area': np.array((0.0, 0.0, w - 1, h - 1)),
            'hflip': False,
            'width_height': np.array((w, h)),
        }
        for k, v in meta_from_image.items():
            if k not in meta:
                meta[k] = v

        return image, anns, meta


class AnnotationJitter(Preprocess):
    def __init__(self, epsilon=0.5):
        self.epsilon = epsilon

    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        for ann in anns:
            keypoints_xy = ann['keypoints'][:, :2]
            sym_rnd_kp = (torch.rand(*keypoints_xy.shape).numpy() - 0.5) * 2.0
            keypoints_xy += self.epsilon * sym_rnd_kp

            sym_rnd_bbox = (torch.rand((4,)).numpy() - 0.5) * 2.0
            ann['bbox'] += 0.5 * self.epsilon * sym_rnd_bbox

        return image, anns, meta

class CenterPad(Preprocess):
    """Pad to a square of given size."""

    def __init__(self, target_size: int):
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_size = target_size

    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        LOG.debug('valid area before pad: %s, image size = %s, target size = %s',
                  meta['valid_area'], image.size, self.target_size)
        image, anns, ltrb = self.center_pad(image, anns)
        meta['offset'] -= ltrb[:2]
        meta['valid_area'][:2] += ltrb[:2]
        LOG.debug('valid area after pad: %s, image size = %s', meta['valid_area'], image.size)

        return image, anns, meta

    def center_pad(self, image, anns):
        w, h = image.size

        left = int((self.target_size[0] - w) / 2.0)
        top = int((self.target_size[1] - h) / 2.0)
        left = max(0, left)
        top = max(0, top)

        right = self.target_size[0] - w - left
        bottom = self.target_size[1] - h - top
        right = max(0, right)
        bottom = max(0, bottom)
        ltrb = (left, top, right, bottom)
        LOG.debug('pad with %s', ltrb)

        # pad image
        fill_value = int(torch.randint(0, 255, (1,)).item())
        image = torchvision.transforms.functional.pad(
            image, ltrb, fill=(fill_value, fill_value, fill_value))

        # pad annotations
        for ann in anns:
            ann['keypoints'][:, 0] += ltrb[0]
            ann['keypoints'][:, 1] += ltrb[1]
            ann['bbox'][0] += ltrb[0]
            ann['bbox'][1] += ltrb[1]

        return image, anns, ltrb


class CenterPadTight(Preprocess):
    def __init__(self, multiple):
        self.multiple = multiple

    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

class Encoders(Preprocess):
    """Preprocess operation that runs encoders."""
    def __init__(self, encoders):
        self.encoders = encoders

    def __call__(self, image, anns, meta):
        anns = [enc(image, anns, meta) for enc in self.encoders]
        meta['head_indices'] = [enc.meta.head_index for enc in self.encoders]
        return image, anns, meta

class ImageTransform(Preprocess):
    """Transform image without modifying annotations or meta."""
    def __init__(self, image_transform):
        self.image_transform = image_transform

    def __call__(self, image, anns, meta):
        image = self.image_transform(image)
        return image, anns, meta


class JpegCompression(Preprocess):
    """Add jpeg compression."""
    def __init__(self, quality=50):
        self.quality = quality

    def __call__(self, image, anns, meta):
        f = io.BytesIO()
        image.save(f, 'jpeg', quality=self.quality)
        return PIL.Image.open(f), anns, meta


class Blur(Preprocess):
    """Blur image."""
    def __init__(self, max_sigma=5.0):
        self.max_sigma = max_sigma

    def __call__(self, image, anns, meta):
        im_np = np.asarray(image)
        sigma = self.max_sigma * float(torch.rand(1).item())
        im_np = scipy.ndimage.filters.gaussian_filter(im_np, sigma=(sigma, sigma, 0))
        return PIL.Image.fromarray(im_np), anns, meta


class HorizontalBlur(Preprocess):
    def __init__(self, sigma=5.0):
        self.sigma = sigma

    def __call__(self, image, anns, meta):
        im_np = np.asarray(image)
        sigma = self.sigma * (0.8 + 0.4 * float(torch.rand(1).item()))
        LOG.debug('horizontal blur with %f', sigma)
        im_np = scipy.ndimage.filters.gaussian_filter1d(im_np, sigma=sigma, axis=1)
        return PIL.Image.fromarray(im_np), anns, meta
class _HorizontalSwap():
    def __init__(self, keypoints, hflip):
        self.keypoints = keypoints
        self.hflip = hflip

        # guarantee hflip is symmetric (left -> right implies right -> left)
        for source, target in list(self.hflip.items()):
            if target in self.hflip:
                assert self.hflip[target] == source
            else:
                LOG.warning('adding %s -> %s', target, source)
                self.hflip[target] = source

    def __call__(self, keypoints):
        target = np.zeros(keypoints.shape)

        for source_i, xyv in enumerate(keypoints):
            source_name = self.keypoints[source_i]
            target_name = self.hflip.get(source_name)
            if target_name:
                target_i = self.keypoints.index(target_name)
            else:
                target_i = source_i
            target[target_i] = xyv

        return target


class HFlip(Preprocess):
    """Horizontally flip image and annotations."""
    def __init__(self, keypoints, hflip):
        self.swap = _HorizontalSwap(keypoints, hflip)

    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        w, _ = image.size
        image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        for ann in anns:
            ann['keypoints'][:, 0] = -ann['keypoints'][:, 0] - 1.0 + w
            if self.swap is not None and not ann['iscrowd']:
                ann['keypoints'] = self.swap(ann['keypoints'])
                meta['horizontal_swap'] = self.swap
            ann['bbox'][0] = -(ann['bbox'][0] + ann['bbox'][2]) - 1.0 + w

        assert meta['hflip'] is False
        meta['hflip'] = True

        meta['valid_area'][0] = -(meta['valid_area'][0] + meta['valid_area'][2]) - 1.0 + w

        return image, anns, meta
class RandomApply(Preprocess):
    """Randomly apply another transformation.

    :param transform: another transformation
    :param probability: probability to apply the given transform
    """
    def __init__(self, transform: Preprocess, probability: float):
        self.transform = transform
        self.probability = probability

    def __call__(self, image, anns, meta):
        if float(torch.rand(1).item()) > self.probability:
            return image, anns, meta
        return self.transform(image, anns, meta)


class RandomChoice(Preprocess):
    """Choose a single random transform."""
    def __init__(self, transforms: List[Preprocess], probabilities: List[float]):
        if sum(probabilities) < 1.0 and len(transforms) == len(probabilities):
            transforms.append(None)
        self.transforms = transforms

        if len(transforms) == len(probabilities) + 1:
            probabilities.append(1.0 - sum(probabilities))
        assert sum(probabilities) == 1.0, [transforms, probabilities]
        assert len(transforms) == len(probabilities)
        self.probabilities = probabilities

    def __call__(self, image, anns, meta):
        rnd = float(torch.rand(1).item())
        for t, p_cumulative in zip(self.transforms, itertools.accumulate(self.probabilities)):
            if rnd > p_cumulative:
                continue

            if t is None:
                return image, anns, meta
            return t(image, anns, meta)

        raise Exception('not possible')

def rotate(image, anns, meta, angle):
    meta = copy.deepcopy(meta)
    anns = copy.deepcopy(anns)

    LOG.debug('rotation angle = %f', angle)
    w, h = image.size
    assert meta['rotation']['angle'] == 0.0
    meta['rotation']['angle'] = angle
    meta['rotation']['width'] = w
    meta['rotation']['height'] = h

    # rotate image
    if angle != 0.0:
        im_np = np.asarray(image)
        if im_np.shape[0] == im_np.shape[1] and angle == 90:
            im_np = np.swapaxes(im_np, 0, 1)
            im_np = np.flip(im_np, axis=0)
        elif im_np.shape[0] == im_np.shape[1] and angle == 270:
            im_np = np.swapaxes(im_np, 0, 1)
            im_np = np.flip(im_np, axis=1)
        elif im_np.shape[0] == im_np.shape[1] and angle == 180:
            im_np = np.flip(im_np, axis=0)
            im_np = np.flip(im_np, axis=1)
        else:
            fill_value = int(torch.randint(0, 255, (1,)).item())
            im_np = scipy.ndimage.rotate(im_np, angle=angle, cval=fill_value, reshape=False)
        image = PIL.Image.fromarray(im_np)
    LOG.debug('rotated by = %f degrees', angle)

    # rotate keypoints
    cangle = math.cos(angle / 180.0 * math.pi)
    sangle = math.sin(angle / 180.0 * math.pi)
    for ann in anns:
        xy = ann['keypoints'][:, :2]
        x_old = xy[:, 0].copy() - (w - 1) / 2
        y_old = xy[:, 1].copy() - (h - 1) / 2
        xy[:, 0] = (w - 1) / 2 + cangle * x_old + sangle * y_old
        xy[:, 1] = (h - 1) / 2 - sangle * x_old + cangle * y_old
        ann['bbox'] = rotate_box(ann['bbox'], w - 1, h - 1, angle)

    LOG.debug('meta before: %s', meta)
    meta['valid_area'] = rotate_box(meta['valid_area'], w - 1, h - 1, angle)
    # fix valid area to be inside original image dimensions
    original_valid_area = meta['valid_area'].copy()
    meta['valid_area'][0] = np.clip(meta['valid_area'][0], 0, w - 1)
    meta['valid_area'][1] = np.clip(meta['valid_area'][1], 0, h - 1)
    new_rb_corner = original_valid_area[:2] + original_valid_area[2:]
    new_rb_corner[0] = np.clip(new_rb_corner[0], 0, w - 1)
    new_rb_corner[1] = np.clip(new_rb_corner[1], 0, h - 1)
    meta['valid_area'][2:] = new_rb_corner - meta['valid_area'][:2]
    LOG.debug('meta after: %s', meta)

    return image, anns, meta


def _prepad(image, anns, meta, angle):
    if abs(angle) < 0.3:
        return image, anns, meta

    w, h = image.size
    cos_angle = math.cos(abs(angle) * math.pi / 180.0)
    sin_angle = math.sin(abs(angle) * math.pi / 180.0)
    LOG.debug('angle = %f, cos = %f, sin = %f', angle, cos_angle, sin_angle)
    padded_size = (
        int(w * cos_angle + h * sin_angle) + 1,
        int(h * cos_angle + w * sin_angle) + 1,
    )
    center_pad = CenterPad(padded_size)
    return center_pad(image, anns, meta)


class RotateBy90(Preprocess):
    """Randomly rotate by multiples of 90 degrees."""

    def __init__(self, angle_perturbation=0.0, fixed_angle=None, prepad=False):
        super().__init__()

        self.angle_perturbation = angle_perturbation
        self.fixed_angle = fixed_angle
        self.prepad = prepad

    def __call__(self, image, anns, meta):
        if self.fixed_angle is not None:
            angle = self.fixed_angle
        else:
            rnd1 = float(torch.rand(1).item())
            angle = int(rnd1 * 4.0) * 90.0
            sym_rnd2 = (float(torch.rand(1).item()) - 0.5) * 2.0
            angle += sym_rnd2 * self.angle_perturbation

        if self.prepad:
            image, anns, meta = _prepad(image, anns, meta, angle)
        return rotate(image, anns, meta, angle)


class RotateUniform(Preprocess):
    """Rotate by a random angle uniformly drawn from a given angle range."""

    def __init__(self, max_angle=30.0, prepad=True):
        super().__init__()
        self.max_angle = max_angle
        self.prepad = prepad

    def __call__(self, image, anns, meta):
        sym_rnd = (float(torch.rand(1).item()) - 0.5) * 2.0
        angle = sym_rnd * self.max_angle

        if self.prepad:
            image, anns, meta = _prepad(image, anns, meta, angle)
        return rotate(image, anns, meta, angle)
def _scale(image, anns, meta, target_w, target_h, resample, *, fast=False):
    """target_w and target_h as integers

    Internally, resample in Pillow are aliases:
    PIL.Image.BILINEAR = 2
    PIL.Image.BICUBIC = 3
    """
    meta = copy.deepcopy(meta)
    anns = copy.deepcopy(anns)
    w, h = image.size

    assert resample in (0, 2, 3)

    # scale image
    if fast and cv2 is not None:
        LOG.debug('using OpenCV for fast rescale')
        if resample == 0:
            cv_interpoltation = cv2.INTER_NEAREST
        elif resample == 2:
            cv_interpoltation = cv2.INTER_LINEAR
        elif resample == 3:
            cv_interpoltation = cv2.INTER_CUBIC
        else:
            raise NotImplementedError('resample of {} not implemented for OpenCV'.format(resample))
        im_np = np.asarray(image)
        im_np = cv2.resize(im_np, (target_w, target_h), interpolation=cv_interpoltation)
        image = PIL.Image.fromarray(im_np)
    elif fast:
        LOG.debug('Requested fast resizing without OpenCV. Using Pillow. '
                  'Install OpenCV for even faster image resizing.')
        image = image.resize((target_w, target_h), resample)
    else:
        order = resample
        if order == 2:
            order = 1

        im_np = np.asarray(image)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            im_np = scipy.ndimage.zoom(im_np, (target_h / h, target_w / w, 1), order=order)
        image = PIL.Image.fromarray(im_np)

    LOG.debug('before resize = (%f, %f), after = %s', w, h, image.size)
    assert image.size[0] == target_w
    assert image.size[1] == target_h

    # rescale keypoints
    x_scale = (image.size[0] - 1) / (w - 1)
    y_scale = (image.size[1] - 1) / (h - 1)
    scale_factors = np.array((x_scale, y_scale))
    for ann in anns:
        ann['keypoints'][:, [0, 1]] *= np.expand_dims(scale_factors, 0)
        ann['bbox'][:2] *= scale_factors
        ann['bbox'][2:] *= scale_factors

    # adjust meta
    LOG.debug('meta before: %s', meta)
    meta['offset'] *= scale_factors
    meta['scale'] *= scale_factors
    meta['valid_area'][:2] *= scale_factors
    meta['valid_area'][2:] *= scale_factors
    LOG.debug('meta after: %s', meta)

    return image, anns, meta


class RescaleRelative(Preprocess):
    """Rescale relative to input image."""

    def __init__(self, scale_range=(0.5, 1.0), *,
                 resample=PIL.Image.BILINEAR,
                 absolute_reference=None,
                 fast=False,
                 power_law=False,
                 stretch_range=None):
        self.scale_range = scale_range
        self.resample = resample
        self.absolute_reference = absolute_reference
        self.fast = fast
        self.power_law = power_law
        self.stretch_range = stretch_range

    def __call__(self, image, anns, meta):
        if isinstance(self.scale_range, tuple):
            if self.power_law:
                rnd_range = np.log2(self.scale_range[0]), np.log2(self.scale_range[1])
                log2_scale_factor = (
                    rnd_range[0]
                    + torch.rand(1).item() * (rnd_range[1] - rnd_range[0])
                )

                scale_factor = 2 ** log2_scale_factor
                LOG.debug('rnd range = %s, log2_scale_Factor = %f, scale factor = %f',
                          rnd_range, log2_scale_factor, scale_factor)
            else:
                scale_factor = (
                    self.scale_range[0]
                    + torch.rand(1).item() * (self.scale_range[1] - self.scale_range[0])
                )
        else:
            scale_factor = self.scale_range

        w, h = image.size
        if self.absolute_reference is not None:
            if w > h:
                h *= self.absolute_reference / w
                w = self.absolute_reference
            else:
                w *= self.absolute_reference / h
                h = self.absolute_reference

        stretch_factor = 1.0
        if self.stretch_range is not None:
            stretch_factor = (
                self.stretch_range[0]
                + torch.rand(1).item() * (self.stretch_range[1] - self.stretch_range[0])
            )

        target_w, target_h = int(w * scale_factor * stretch_factor), int(h * scale_factor)
        return _scale(image, anns, meta, target_w, target_h, self.resample, fast=self.fast)


class Crop(Preprocess):
    """Random cropping."""

    def __init__(self, long_edge, use_area_of_interest=True):
        self.long_edge = long_edge
        self.use_area_of_interest = use_area_of_interest

    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)
        original_valid_area = meta['valid_area'].copy()

        image, anns, ltrb = self.crop(image, anns, meta['valid_area'])
        meta['offset'] += ltrb[:2]

        new_wh = image.size
        LOG.debug('valid area before crop of %s: %s', ltrb, original_valid_area)
        # process crops from left and top
        meta['valid_area'][:2] = np.maximum(0.0, original_valid_area[:2] - ltrb[:2])
        # process crops from right and bottom
        new_rb_corner = original_valid_area[:2] + original_valid_area[2:] - ltrb[:2]
        new_rb_corner = np.maximum(meta['valid_area'][:2], new_rb_corner)
        new_rb_corner = np.minimum(new_wh, new_rb_corner)
        meta['valid_area'][2:] = new_rb_corner - meta['valid_area'][:2]
        LOG.debug('valid area after crop: %s', meta['valid_area'])

        # clip bounding boxes
        for ann in anns:
            unclipped_bbox = ann['bbox'].copy()
            ann['bbox'][:2] = np.maximum(meta['valid_area'][:2], ann['bbox'][:2])
            new_rb = unclipped_bbox[:2] + unclipped_bbox[2:]
            new_rb = np.maximum(ann['bbox'][:2], new_rb)
            new_rb = np.minimum(meta['valid_area'][:2] + meta['valid_area'][2:], new_rb)
            ann['bbox'][2:] = new_rb - ann['bbox'][:2]
        anns = [ann for ann in anns if ann['bbox'][2] > 0.0 and ann['bbox'][3] > 0.0]

        return image, anns, meta

    @staticmethod
    def area_of_interest(anns, valid_area):
        """area that contains annotations with keypoints"""

        points_of_interest = [
            xy
            for ann in anns
            if not ann.get('iscrowd', False)
            for xy in [ann['bbox'][:2], ann['bbox'][:2] + ann['bbox'][2:]]
        ]
        if not points_of_interest:
            return valid_area
        points_of_interest = np.stack(points_of_interest, axis=0)
        min_xy = np.min(points_of_interest, axis=0) - 50
        max_xy = np.max(points_of_interest, axis=0) + 50

        # Make sure to stay inside of valid area.
        left = np.clip(min_xy[0], valid_area[0], valid_area[0] + valid_area[2] - 1)
        top = np.clip(min_xy[1], valid_area[1], valid_area[1] + valid_area[3] - 1)
        right = np.clip(max_xy[0], left + 1, valid_area[0] + valid_area[2])
        bottom = np.clip(max_xy[1], top + 1, valid_area[1] + valid_area[3])

        return (left, top, right - left, bottom - top)

    @staticmethod
    def random_location_1d(image_length,
                           valid_min, valid_length,
                           interest_min, interest_length,
                           crop_length,
                           tail=0.1, shift=0.0, fix_inconsistent=True):
        if image_length <= crop_length:
            return 0

        if fix_inconsistent:
            # relevant for tracking with inconsistent image sizes
            # (e.g. with RandomizeOneFrame augmentation)
            valid_min = np.clip(valid_min, 0, image_length)
            valid_length = np.clip(valid_length, 0, image_length - valid_min)
            interest_min = np.clip(interest_min, 0, image_length)
            interest_length = np.clip(interest_length, 0, image_length - interest_min)

        sticky_rnd = -tail + 2 * tail * torch.rand((1,)).item()
        sticky_rnd = np.clip(sticky_rnd, 0.0, 1.0)

        if interest_length > crop_length:
            # crop within area of interest
            sticky_rnd = np.clip(sticky_rnd + shift / interest_length, 0.0, 1.0)
            offset = interest_min + (interest_length - crop_length) * sticky_rnd
            return int(offset)

        # from above: interest_length < crop_length
        min_v = interest_min + interest_length - crop_length
        max_v = interest_min

        if valid_length > crop_length:
            # clip to valid area
            min_v = max(min_v, valid_min)
            max_v = max(min_v, min(max_v, valid_min + valid_length - crop_length))
        elif image_length > crop_length:
            # clip to image
            min_v = max(min_v, 0)
            max_v = max(min_v, min(max_v, 0 + image_length - crop_length))

        # image constraint
        min_v = np.clip(min_v, 0, image_length - crop_length)
        max_v = np.clip(max_v, 0, image_length - crop_length)

        assert max_v >= min_v
        sticky_rnd = np.clip(sticky_rnd + shift / (max_v - min_v + 1e-3), 0.0, 1.0)
        offset = min_v + (max_v - min_v) * sticky_rnd
        return int(offset)

    def crop(self, image, anns, valid_area):
        if self.use_area_of_interest:
            area_of_interest = self.area_of_interest(anns, valid_area)
        else:
            area_of_interest = valid_area

        w, h = image.size
        x_offset, y_offset = 0, 0
        if w > self.long_edge:
            x_offset = self.random_location_1d(
                w - 1,
                valid_area[0], valid_area[2],
                area_of_interest[0], area_of_interest[2],
                self.long_edge,
            )
        if h > self.long_edge:
            y_offset = self.random_location_1d(
                h - 1,
                valid_area[1], valid_area[3],
                area_of_interest[1], area_of_interest[3],
                self.long_edge
            )
        LOG.debug('crop offsets (%d, %d)', x_offset, y_offset)

        # crop image
        new_w = min(self.long_edge, w - x_offset)
        new_h = min(self.long_edge, h - y_offset)
        # ltrb might be confusing name:
        # it's the coordinates of the top-left corner and the coordinates
        # of the bottom right corner
        ltrb = (x_offset, y_offset, x_offset + new_w, y_offset + new_h)
        image = image.crop(ltrb)

        # crop keypoints
        for ann in anns:
            ann['keypoints'][:, 0] -= x_offset
            ann['keypoints'][:, 1] -= y_offset
            ann['bbox'][0] -= x_offset
            ann['bbox'][1] -= y_offset

        return image, anns, np.array(ltrb)



class RescaleAbsolute(Preprocess):
    """Rescale to a given size."""

    def __init__(self, long_edge, *, fast=False, resample=PIL.Image.BILINEAR):
        self.long_edge = long_edge
        self.fast = fast
        self.resample = resample

    def __call__(self, image, anns, meta):
        w, h = image.size

        this_long_edge = self.long_edge
        if isinstance(this_long_edge, (tuple, list)):
            this_long_edge = torch.randint(
                int(this_long_edge[0]),
                int(this_long_edge[1]), (1,)
            ).item()

        s = this_long_edge / max(h, w)
        if h > w:
            target_w, target_h = int(w * s), int(this_long_edge)
        else:
            target_w, target_h = int(this_long_edge), int(h * s)
        return _scale(image, anns, meta, target_w, target_h, self.resample, fast=self.fast)


class ScaleMix(Preprocess):
    def __init__(self, scale_threshold, *,
                 upscale_factor=2.0,
                 downscale_factor=0.5,
                 resample=PIL.Image.BILINEAR):
        self.scale_threshold = scale_threshold
        self.upscale_factor = upscale_factor
        self.downscale_factor = downscale_factor
        self.resample = resample

    def __call__(self, image, anns, meta):
        scales = np.array([
            np.sqrt(ann['bbox'][2] * ann['bbox'][3])
            for ann in anns if (not getattr(ann, 'iscrowd', False)
                                and np.any(ann['keypoints'][:, 2] > 0.0))
        ])
        LOG.debug('scale threshold = %f, scales = %s', self.scale_threshold, scales)
        if not scales.shape[0]:
            return image, anns, meta

        all_above_threshold = np.all(scales > self.scale_threshold)
        all_below_threshold = np.all(scales < self.scale_threshold)
        if not all_above_threshold and \
           not all_below_threshold:
            return image, anns, meta

        w, h = image.size
        if all_above_threshold:
            target_w, target_h = int(w / 2), int(h / 2)
        else:
            target_w, target_h = int(w * 2), int(h * 2)
        LOG.debug('scale mix from (%d, %d) to (%d, %d)', w, h, target_w, target_h)
        return _scale(image, anns, meta, target_w, target_h, self.resample)

EVAL_TRANSFORM = Compose([
    NormalizeAnnotations(),
    ImageTransform(torchvision.transforms.ToTensor()),
    ImageTransform(
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
    ),
])


TRAIN_TRANSFORM = Compose([
    NormalizeAnnotations(),
    ImageTransform(torchvision.transforms.ColorJitter(
        brightness=0.4, contrast=0.1, saturation=0.4, hue=0.1)),
    RandomApply(JpegCompression(), 0.1),  # maybe irrelevant for COCO, but good for others
    # RandomApply(Blur(), 0.01),  # maybe irrelevant for COCO, but good for others
    ImageTransform(torchvision.transforms.RandomGrayscale(p=0.01)),
    EVAL_TRANSFORM,
])

class ToAnnotations(Preprocess):
    """Convert inputs to annotation objects."""

    def __init__(self, converters):
        self.converters = converters

    def __call__(self, image, anns, meta):
        anns = [
            ann
            for converter in self.converters
            for ann in converter(anns)
        ]
        return image, anns, meta


class ToKpAnnotations:
    """Input to keypoint annotations."""

    def __init__(self, categories, keypoints_by_category, skeleton_by_category):
        self.keypoints_by_category = keypoints_by_category
        self.skeleton_by_category = skeleton_by_category
        self.categories = categories

    def __call__(self, anns):
        return [
            Annotation(
                self.keypoints_by_category[ann['category_id']],
                self.skeleton_by_category[ann['category_id']],
                categories=self.categories,
            )
            .set(
                ann['keypoints'],
                category_id=ann['category_id'],
                fixed_score='',
                fixed_bbox=ann.get('bbox'),
            )
            for ann in anns
            if not ann['iscrowd'] and np.any(ann['keypoints'][2::3] > 0.0)
        ]


class ToDetAnnotations:
    """Input to detection annotations."""

    def __init__(self, categories):
        self.categories = categories

    def __call__(self, anns):
        return [
            AnnotationDet(categories=self.categories)
            .set(
                ann['category_id'],
                None,
                ann['bbox'],
            )
            for ann in anns
            if not ann['iscrowd'] and np.any(ann['bbox'])
        ]


class ToCrowdAnnotations:
    """Input to crowd annotations."""

    def __init__(self, categories):
        self.categories = categories

    def __call__(self, anns):
        return [
            AnnotationCrowd(categories=self.categories)
            .set(
                ann.get('category_id', 1),
                ann['bbox'],
            )
            for ann in anns
            if ann['iscrowd']
        ]