from dataloader import *
from model import *
from visualizer import Base, CifVisualizer, CafVisualizer, CifHr
from heads import Cif, Caf
from model import get_coco_multihead
from annotation import Annotation
import pycocotools.coco
from pycocotools.cocoeval import COCOeval

def processor(model, processed_image_batch, device):
	fbatch = fields_batch(model, processed_image_batch, device=device)
	fbatch = decode(fbatch[0])
	result = Dpool.starmap(mappable_annotations, zip(fbatch, processed_image_batch, [None for _ in fbatch]))
	return result


def enumerated_dataloader(enumerated_dataloader, model, device):
    for batch_i, item in enumerated_dataloader:
        if len(item) == 3:
            processed_image_batch, gt_anns_batch, meta_batch = item
            image_batch = [None for _ in processed_image_batch]
        elif len(item) == 4:
            image_batch, processed_image_batch, gt_anns_batch, meta_batch = item

        pred_batch = processor(model, processed_image_batch, device=device)


        # un-batch
        for image, pred, gt_anns, meta in zip(image_batch, pred_batch, gt_anns_batch, meta_batch):
            pred = [pred]
            gt_anns = [gt_anns]


            yield pred, gt_anns, meta

def accumulate(predictions, image_meta, *, ground_truth=None):
    image_id = int(image_meta['image_id'])
    # self.image_ids.append(image_id)

    if args.threshold:
        predictions = [pred for pred in predictions
                       if pred.scale(v_th=0.01) >= args.threshold]
    if len(predictions) > args.max_per_image:
        predictions = predictions[:args.max_per_image]

    image_annotations = []
    for pred in predictions:
        pred_data = pred.json_data()
        pred_data['image_id'] = image_id
        pred_data = {
            k: v for k, v in pred_data.items()
            if k in ('category_id', 'score', 'keypoints', 'bbox', 'image_id')
        }
        image_annotations.append(pred_data)
    if not image_annotations:
        n_keypoints = 17
        image_annotations.append({
            'image_id': image_id,
            'category_id': 1,
            'keypoints': np.zeros((n_keypoints * 3,)).tolist(),
            'bbox': [0, 0, 1, 1],
            'score': 0.001,
        })

    predictions += image_annotations
    return predictions

def fields_batch(model, image_batch, device=None):

    def apply(f, items):
        if items is None:
            return None
        if isinstance(items, (list, tuple)):
            return [apply(f, i) for i in items]
        return f(items)

    with torch.no_grad():
        if device is not None:
            image_batch = image_batch.to(device, non_blocking=True)

        with torch.autograd.profiler.record_function('model'):
            heads = model(image_batch)

        # to numpy
        with torch.autograd.profiler.record_function('tonumpy'):
            heads = apply(lambda x: x.cpu(), heads)

    # index by frame (item in batch)
    head_iter = apply(iter, heads)
    heads = []
    while True:
        try:
            heads.append(apply(next, head_iter))
        except StopIteration:
            break

    return heads

class Dpool():
    @staticmethod
    def starmap(f, iterable):
        return [f(*i) for i in iterable]

def mappable_annotations(fields, debug_image, gt_anns):
    if debug_image is not None:
        Base.processed_image(debug_image)
    if gt_anns is not None:
        Base.ground_truth(gt_anns)

    return fields

def decode(fields, initial_annotations=None):
	headsmeta = get_coco_multihead()
	cif_metas = [headsmeta[0]]
	cif_visualizers = [CifVisualizer(m) for m in cif_metas]
	caf_metas = [headsmeta[1]]
	caf_visualizers = [CafVisualizer(m) for m in caf_metas]
	cifhr_visualizers = [
                CifHr(stride=meta.stride, field_names=meta.keypoints)
                for meta in cif_metas
            ]
	if not initial_annotations:
	    initial_annotations_t = None
	    initial_ids_t = None
	else:
	    initial_annotations_t = torch.empty(
	        (len(initial_annotations), cif_metas[0].n_fields, 4))
	    initial_ids_t = torch.empty((len(initial_annotations),), dtype=torch.int64)
	    for i, (ann_py, ann_t) in enumerate(zip(initial_annotations, initial_annotations_t)):
	        for f in range(len(ann_py.data)):
	            ann_t[f, 0] = float(ann_py.data[f, 2])
	            ann_t[f, 1] = float(ann_py.data[f, 0])
	            ann_t[f, 2] = float(ann_py.data[f, 1])
	            ann_t[f, 3] = float(ann_py.joint_scales[f])
	        initial_ids_t[i] = getattr(ann_py, 'id_', -1)
	    LOG.debug('initial annotations = %d', initial_annotations_t.size(0))

	for vis, meta in zip(cif_visualizers, cif_metas):
		vis.predicted(fields[0])
	for vis, meta in zip(caf_visualizers, caf_metas):
	    vis.predicted(fields[1])
	cpp_decoder = torch.classes.openpifpaf_decoder.CifCaf(
            len(cif_metas[0].keypoints),
            torch.LongTensor(caf_metas[0].skeleton) - 1,
        )
	annotations, annotation_ids = cpp_decoder.call_with_initial_annotations(
	    fields[0],
	    cif_metas[0].stride,
	    fields[1],
	    caf_metas[0].stride,
	    initial_annotations_t,
	    initial_ids_t,
	)
	for vis in cifhr_visualizers:
	    fields, low = cpp_decoder.get_cifhr()
	    vis.predicted(fields, low)
	annotations_py = []
	score_weights = cif_metas[0].score_weights
	for ann_data, ann_id in zip(annotations, annotation_ids):

	    ann = Annotation(cif_metas[0].keypoints,
	                     caf_metas[0].skeleton,
	                     score_weights=score_weights)
	    ann.data[:, :2] = ann_data[:, 1:3]
	    ann.data[:, 2] = ann_data[:, 0]
	    ann.joint_scales[:] = ann_data[:, 3]
	    if ann_id != -1:
	        ann.id_ = int(ann_id)
	    annotations_py.append(ann)

	LOG.info('annotations %d: %s',
	         len(annotations_py),
	         [np.sum(ann.data[:, 2] > 0.1) for ann in annotations_py])
	return annotations_py



def main():
	test_loader = Cocodata(args).eval_loader()
	checkpoint = torch.load(args.checkpoint)
	model = checkpoint['model'].cuda()
	model.eval()
	device = torch.device('cuda')
	pred_loader = enumerated_dataloader(enumerate(test_loader), model, device)
	predictions = []
	image_ids = []
	for image_i, (pred, gt_anns, image_meta) in enumerate(pred_loader):
		image_ids.append(image_i)
		predictions += accumulate(pred, image_meta, ground_truth = gt_anns)


	predictions = [
            {k: v for k, v in annotation.items()
             if k in ('image_id', 'category_id', 'keypoints', 'score')}
            for annotation in predictions
            if type(annotation) == dict
        ]
	coco = pycocotools.coco.COCO(annotation_file='./data-mscoco/annotations/person_keypoints_val2017.json')
	coco_eval = coco.loadRes(predictions)

	myeval = COCOeval(coco, coco_eval, iouType='keypoints')
	category_ids = [1]
	if category_ids:
	    myeval.params.catIds = category_ids

	if image_ids is not None:
	    print('image ids', image_ids)
	    myeval.params.imgIds = image_ids
	myeval.evaluate()
	myeval.accumulate()
	myeval.summarize()
	print(myeval.stats)
	    

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	# parser.add_argument('--version', action='version',
	#                     version='OpenPifPaf {version}'.format(version=__version__))
	parser.add_argument('-c', '--checkpoint', default='output/shufflenet-test.epoch022',
	                    help='output file')
	parser.add_argument('--batch-size', default=1, type=int)
	parser.add_argument('--loader_workers', default=4, type=int)
	parser.add_argument('--threshold', default=0.5, type=float)
	parser.add_argument('--max_per_image', default=20, type=int)
	args = parser.parse_args()
	main()
