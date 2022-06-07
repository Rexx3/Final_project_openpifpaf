from dataloader import *
from model import *
from visualizer import Base

def enumerated_dataloader(enumerated_dataloader):
    """Predict from an enumerated dataloader."""
    for batch_i, item in enumerated_dataloader:
        if len(item) == 3:
            processed_image_batch, gt_anns_batch, meta_batch = item
            image_batch = [None for _ in processed_image_batch]
        elif len(item) == 4:
            image_batch, processed_image_batch, gt_anns_batch, meta_batch = item
        if self.visualize_processed_image:
            visualizer.Base.processed_image(processed_image_batch[0])

        pred_batch = self.processor.batch(self.model, processed_image_batch, device=self.device)
        # self.last_decoder_time = self.processor.last_decoder_time
        # self.last_nn_time = self.processor.last_nn_time
        # self.total_decoder_time += self.processor.last_decoder_time
        # self.total_nn_time += self.processor.last_nn_time
        # self.total_images += len(processed_image_batch)

        # un-batch
        for image, pred, gt_anns, meta in \
                zip(image_batch, pred_batch, gt_anns_batch, meta_batch):
            LOG.info('batch %d: %s', batch_i, meta.get('file_name', 'no-file-name'))

            # load the original image if necessary
            # if self.visualize_image:
            #     visualizer.Base.image(image, meta=meta)

            pred = [ann.inverse_transform(meta) for ann in pred]
            gt_anns = [ann.inverse_transform(meta) for ann in gt_anns]

            # if self.json_data:
            #     pred = [ann.json_data() for ann in pred]

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

    # force at least one annotation per image (for pycocotools)
    if not image_annotations:
        n_keypoints = 17
        image_annotations.append({
            'image_id': image_id,
            'category_id': 1,
            'keypoints': np.zeros((n_keypoints * 3,)).tolist(),
            'bbox': [0, 0, 1, 1],
            'score': 0.001,
        })

    # if LOG.getEffectiveLevel() == logging.DEBUG:
    #     self._stats(image_annotations, [image_id])
    #     LOG.debug(image_meta)

    self.predictions += image_annotations

def fields_batch(model, image_batch, device=None):

    def apply(f, items):
        """Apply f in a nested fashion to all items that are not list or tuple."""
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

def main():
	test_loader = Cocodata(args).eval_loader()
	checkpoint = torch.load(args.checkpoint)
	model = checkpoint['model'].cuda()
	model.eval()
	device = torch.device('cuda')
	for batch_i, item in enumerate(test_loader):
		fbatch = fields_batch(model, item[0], device=device)
		for f in fbatch:
			img = Base.processed_image(f)
			print(type(img))
		print(item[1].inverse_transform(item[2][0]))
		metric = accumulate(item[0][0], item[2][0], ground_truth = item[1])
	    

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
