import torch 
import os
import argparse
from model import *
from heads import *
from heads import Cif
from heads import Caf
import logging
import copy
from PIL import Image
import transforms
import encoder
LOG = logging.getLogger(__name__)

COCO_CATEGORIES = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'street sign',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'hat',
    'backpack',
    'umbrella',
    'shoe',
    'eye glasses',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'plate',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'mirror',
    'dining table',
    'window',
    'desk',
    'toilet',
    'door',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'blender',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
    'hair brush',
]

def collate_images_anns_meta(batch):
    anns = [b[-2] for b in batch]
    metas = [b[-1] for b in batch]

    if len(batch[0]) == 4:
        # raw images are also in this batch
        images = [b[0] for b in batch]
        processed_images = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
        return images, processed_images, anns, metas

    processed_images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    return processed_images, anns, metas


def collate_images_targets_meta(batch):
    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    targets = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
    metas = [b[2] for b in batch]
    return images, targets, metas


def collate_tracking_images_targets_meta(batch):
    images = torch.utils.data.dataloader.default_collate([
        im for group in batch for im in group[0]])

    targets = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
    metas = [b[2] for b in batch]

    return images, targets, metas


class CocoDataset(torch.utils.data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        image_dir (string): Root directory where images are downloaded to.
        ann_file (string): Path to json annotation file.
    """

    def __init__(self, image_dir, ann_file, *,
                 preprocess=None, min_kp_anns=0,
                 category_ids=None,
                 annotation_filter=False):
        super().__init__()

        if category_ids is None:
            category_ids = []

        from pycocotools.coco import COCO  # pylint: disable=import-outside-toplevel
        self.image_dir = image_dir
        self.coco = COCO(ann_file)

        self.category_ids = category_ids

        self.ids = self.coco.getImgIds(catIds=self.category_ids)
        if annotation_filter:
            self.filter_for_annotations(min_kp_anns=min_kp_anns)
        elif min_kp_anns:
            raise Exception('only set min_kp_anns with annotation_filter')
        LOG.info('Images: %d', len(self.ids))

        self.preprocess = preprocess or openpifpaf.transforms.EVAL_TRANSFORM

    def filter_for_annotations(self, *, min_kp_anns=0):
        LOG.info('filter for annotations (min kp=%d) ...', min_kp_anns)

        def filter_image(image_id):
            ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.category_ids)
            anns = self.coco.loadAnns(ann_ids)
            anns = [ann for ann in anns if not ann.get('iscrowd')]
            if not anns:
                return False
            kp_anns = [ann for ann in anns
                       if 'keypoints' in ann and any(v > 0.0 for v in ann['keypoints'][2::3])]
            return len(kp_anns) >= min_kp_anns

        self.ids = [image_id for image_id in self.ids if filter_image(image_id)]
        LOG.info('... done.')

    def class_aware_sample_weights(self, max_multiple=10.0):
        """Class aware sampling.

        To be used with PyTorch's WeightedRandomSampler.

        Reference: Solution for Large-Scale Hierarchical Object Detection
        Datasets with Incomplete Annotation and Data Imbalance
        Yuan Gao, Xingyuan Bu, Yang Hu, Hui Shen, Ti Bai, Xubin Li and Shilei Wen
        """
        ann_ids = self.coco.getAnnIds(imgIds=self.ids, catIds=self.category_ids)
        anns = self.coco.loadAnns(ann_ids)

        category_image_counts = defaultdict(int)
        image_categories = defaultdict(set)
        for ann in anns:
            if ann['iscrowd']:
                continue
            image = ann['image_id']
            category = ann['category_id']
            if category in image_categories[image]:
                continue
            image_categories[image].add(category)
            category_image_counts[category] += 1

        weights = [
            sum(
                1.0 / category_image_counts[category_id]
                for category_id in image_categories[image_id]
            )
            for image_id in self.ids
        ]
        min_w = min(weights)
        LOG.debug('Class Aware Sampling: minW = %f, maxW = %f', min_w, max(weights))
        max_w = min_w * max_multiple
        weights = [min(w, max_w) for w in weights]
        LOG.debug('Class Aware Sampling: minW = %f, maxW = %f', min_w, max(weights))

        return weights

    def __getitem__(self, index):
        image_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.category_ids)
        anns = self.coco.loadAnns(ann_ids)
        anns = copy.deepcopy(anns)

        image_info = self.coco.loadImgs(image_id)[0]
        LOG.debug(image_info)
        local_file_path = os.path.join(self.image_dir, image_info['file_name'])
        with open(local_file_path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        meta = {
            'dataset_index': index,
            'image_id': image_id,
            'file_name': image_info['file_name'],
            'local_file_path': local_file_path,
        }

        if 'flickr_url' in image_info:
            _, flickr_file_name = image_info['flickr_url'].rsplit('/', maxsplit=1)
            flickr_id, _ = flickr_file_name.split('_', maxsplit=1)
            meta['flickr_full_page'] = 'http://flickr.com/photo.gne?id={}'.format(flickr_id)

        # preprocess image and annotations
        image, anns, meta = self.preprocess(image, anns, meta)

        LOG.debug(meta)

        # TODO: convert into transform
        # # log stats
        # for ann in anns:
        #     if getattr(ann, 'iscrowd', False):
        #         continue
        #     if not np.any(ann['keypoints'][:, 2] > 0.0):
        #         continue
        #     STAT_LOG.debug({'bbox': [int(v) for v in ann['bbox']]})

        return image, anns, meta

    def __len__(self):
    	return len(self.ids)

class Cocodata:
	def __init__(self, args):
		# super().__init__(**kwargs)
		path = "C:/Users/wx036/Desktop/UCSD/ECE228/finalproject/"
		self._test2017_annotations = './data-mscoco/annotations/image_info_test2017.json'
		self._testdev2017_annotations = './data-mscoco/annotations/image_info_test-dev2017.json'
		self._test2017_image_dir = './data-mscoco/images/test2017/'

		# cli configurable
		self.train_annotations = path + 'data-mscoco/annotations/person_keypoints_train2017.json'
		self.val_annotations = path +  'data-mscoco/annotations/person_keypoints_val2017.json'
		self.eval_annotations = self.val_annotations
		self.train_image_dir = path +  'data-mscoco/images/train2017/'
		self.val_image_dir = path +  'data-mscoco/images/val2017/'
		self.eval_image_dir = self.val_image_dir

		self.square_edge = 385
		self.with_dense = False
		self.extended_scale = False
		self.orientation_invariant = 0.0
		self.blur = 0.0
		self.augmentation = True
		self.rescale_images = 1.0
		self.upsample_stride = 1
		self.min_kp_anns = 1
		self.bmin = 0.1

		self.eval_annotation_filter = True
		self.eval_long_edge = 641
		self.eval_orientation_invariant = 0.0
		self.eval_extended_scale = False
		self.head_metas = get_coco_multihead()
		self.batch_size = args.batch_size
		self.pin_memory = True
		self.loader_workers = args.loader_workers

	def _preprocess(self):
		encoders = [encoder.Cif(self.head_metas[0], bmin=self.bmin),
		            encoder.Caf(self.head_metas[1], bmin=self.bmin)]
		# if len(self.head_metas) > 2:
		#     encoders.append(openpifpaf.encoder.Caf(self.head_metas[2], bmin=self.bmin))

		if not self.augmentation:
		    return transforms.Compose([
		       	transforms.NormalizeAnnotations(),
		        transforms.RescaleAbsolute(self.square_edge),
		        transforms.CenterPad(self.square_edge),
		        transforms.EVAL_TRANSFORM,
		        transforms.Encoders(encoders),
		    ])

		if self.extended_scale:
		    rescale_t = transforms.RescaleRelative(
		        scale_range=(0.25 * self.rescale_images,
		                     2.0 * self.rescale_images),
		        power_law=True, stretch_range=(0.75, 1.33))
		else:
		    rescale_t = transforms.RescaleRelative(
		        scale_range=(0.4 * self.rescale_images,
		                     2.0 * self.rescale_images),
		        power_law=True, stretch_range=(0.75, 1.33))

		return transforms.Compose([
		    transforms.NormalizeAnnotations(),
		    transforms.RandomApply(
		        transforms.HFlip(COCO_KEYPOINTS, HFLIP), 0.5),
		    rescale_t,
		    transforms.RandomApply(
		        transforms.Blur(), self.blur),
		    transforms.RandomChoice(
		        [transforms.RotateBy90(),
		         transforms.RotateUniform(30.0)],
		        [self.orientation_invariant, 0.4],
		    ),
		    transforms.Crop(self.square_edge, use_area_of_interest=True),
		    transforms.CenterPad(self.square_edge),
		    transforms.TRAIN_TRANSFORM,
		    transforms.Encoders(encoders),
		])
	def _eval_preprocess(self):
		return transforms.Compose([
		# *self.common_eval_preprocess(),
		transforms.ToAnnotations([
		transforms.ToKpAnnotations(
		COCO_CATEGORIES,
		keypoints_by_category={1: self.head_metas[0].keypoints},
		skeleton_by_category={1: self.head_metas[1].skeleton},
		),
		transforms.ToCrowdAnnotations(COCO_CATEGORIES),
		]),
		transforms.EVAL_TRANSFORM,
		])
	def train_loader(self):
	    train_data = CocoDataset(
	        image_dir=self.train_image_dir,
	        ann_file=self.train_annotations,
	        preprocess=self._preprocess(),
	        annotation_filter=True,
	        min_kp_anns=self.min_kp_anns,
	        category_ids=[1],
	    )
	    return torch.utils.data.DataLoader(
	        train_data, batch_size=self.batch_size, shuffle=not self.augmentation,
	        pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
	        collate_fn=collate_images_targets_meta)

	def val_loader(self):
	    val_data = CocoDataset(
	        image_dir=self.val_image_dir,
	        ann_file=self.val_annotations,
	        preprocess=self._preprocess(),
	        annotation_filter=True,
	        min_kp_anns=self.min_kp_anns,
	        category_ids=[1],
	    )
	    return torch.utils.data.DataLoader(
	        val_data, batch_size=self.batch_size, shuffle=not self.augmentation,
	        pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
	        collate_fn=collate_images_targets_meta)
	def eval_loader(self):
	    eval_data = CocoDataset(
	        image_dir=self.eval_image_dir,
	        ann_file=self.eval_annotations,
	        preprocess=self._preprocess(),
	        annotation_filter=self.eval_annotation_filter,
	        min_kp_anns=self.min_kp_anns if self.eval_annotation_filter else 0,
	        category_ids=[1] if self.eval_annotation_filter else [],
	    )
	    return torch.utils.data.DataLoader(
	        eval_data, batch_size=self.batch_size, shuffle=False,
	        pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=False,
	        collate_fn=collate_images_anns_meta)

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', default=1, type=int)
	parser.add_argument('--loader_workers', default=4, type=int)
    # parser.add_argument('-o', '--output', default=None,
    #                     help='output file')
    # parser.add_argument('--disable-cuda', action='store_true',
    #                     help='disable CUDA')
    # parser.add_argument('--ddp', default=False, action='store_true',
    #                     help='[experimental] DistributedDataParallel')
    # parser.add_argument('--local_rank', default=None, type=int,
    #                     help='[experimental] for torch.distributed.launch')
    # parser.add_argument('--no-sync-batchnorm', dest='sync_batchnorm',
    #                     default=True, action='store_false',
    #                     help='[experimental] in ddp, to not use syncbatchnorm')
	args = parser.parse_args()
	# train_loader = Cocodata(args).train_loader()
	# val_loader = Cocodata(args).train_loader()
	test_loader = Cocodata(args).eval_loader()
	print(len(test_loader))