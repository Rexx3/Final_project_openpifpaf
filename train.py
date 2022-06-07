import argparse
from model import *
from computeloss import build_loss
from heads import *
import torch
import copy
from dataloader import *
from trainer import *

class LearningRateLambda():
    def __init__(self, decay_schedule, *,
                 decay_factor=0.1,
                 decay_epochs=1.0,
                 warm_up_start_epoch=0,
                 warm_up_epochs=2.0,
                 warm_up_factor=0.01,
                 warm_restart_schedule=None,
                 warm_restart_duration=0.5):
        self.decay_schedule = decay_schedule
        self.decay_factor = decay_factor
        self.decay_epochs = decay_epochs
        self.warm_up_start_epoch = warm_up_start_epoch
        self.warm_up_epochs = warm_up_epochs
        self.warm_up_factor = warm_up_factor
        self.warm_restart_schedule = warm_restart_schedule
        self.warm_restart_duration = warm_restart_duration

    def __call__(self, step_i):
        lambda_ = 1.0

        if step_i <= self.warm_up_start_epoch:
            lambda_ *= self.warm_up_factor
        elif self.warm_up_start_epoch < step_i < self.warm_up_start_epoch + self.warm_up_epochs:
            lambda_ *= self.warm_up_factor**(
                1.0 - (step_i - self.warm_up_start_epoch) / self.warm_up_epochs
            )

        for d in self.decay_schedule:
            if step_i >= d + self.decay_epochs:
                lambda_ *= self.decay_factor
            elif step_i > d:
                lambda_ *= self.decay_factor**(
                    (step_i - d) / self.decay_epochs
                )

        for r in self.warm_restart_schedule:
            if r <= step_i < r + self.warm_restart_duration:
                lambda_ = lambda_**(
                    (step_i - r) / self.warm_restart_duration
                )

        return lambda_

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	# parser.add_argument('--version', action='version',
	#                     version='OpenPifPaf {version}'.format(version=__version__))
	parser.add_argument('-o', '--output', default=None,
	                    help='output file')
	parser.add_argument('--disable-cuda', action='store_true',
	                    help='disable CUDA')
	parser.add_argument('--ddp', default=False, action='store_true',
	                    help='[experimental] DistributedDataParallel')
	parser.add_argument('--local_rank', default=None, type=int,
	                    help='[experimental] for torch.distributed.launch')
	parser.add_argument('--no-sync-batchnorm', dest='sync_batchnorm',
	                    default=True, action='store_false',
	                    help='[experimental] in ddp, to not use syncbatchnorm')
	parser.add_argument('--momentum', type=float, default=0.9,
	                   help='SGD momentum, beta1 in Adam')
	parser.add_argument('--beta2', type=float, default=0.999,
	                   help='beta2 for Adam/AMSGrad')
	parser.add_argument('--adam-eps', type=float, default=1e-6,
	                   help='eps value for Adam/AMSGrad')
	parser.add_argument('--no-nesterov', dest='nesterov', default=True, action='store_false',
	                   help='do not use Nesterov momentum for SGD update')
	parser.add_argument('--weight-decay', type=float, default=0.0,
	                   help='SGD/Adam/AMSGrad weight decay')
	parser.add_argument('--adam', action='store_true',
	                   help='use Adam optimizer')
	parser.add_argument('--amsgrad', action='store_true',
	                   help='use Adam optimizer with AMSGrad option')

	parser.add_argument_group('learning rate scheduler')
	parser.add_argument('--lr', type=float, default=1e-3,
	                     help='learning rate')
	parser.add_argument('--lr-decay', default=[], nargs='+', type=float,
	                     help='epochs at which to decay the learning rate')
	parser.add_argument('--lr-decay-factor', default=0.1, type=float,
	                     help='learning rate decay factor')
	parser.add_argument('--lr-decay-epochs', default=1.0, type=float,
	                     help='learning rate decay duration in epochs')
	parser.add_argument('--lr-warm-up-start-epoch', default=0, type=float,
	                     help='starting epoch for warm-up')
	parser.add_argument('--lr-warm-up-epochs', default=1, type=float,
	                     help='number of epochs at the beginning with lower learning rate')
	parser.add_argument('--lr-warm-up-factor', default=0.001, type=float,
	                     help='learning pre-factor during warm-up')
	parser.add_argument('--lr-warm-restarts', default=[], nargs='+', type=float,
	                     help='list of epochs to do a warm restart')
	parser.add_argument('--lr-warm-restart-duration', default=0.5, type=float,
	                     help='duration of a warm restart')

	parser.add_argument('--batch_size', default=1, type=int)
	parser.add_argument('--loader_workers', default=4, type=int)


	Trainer.cli(parser)
	args = parser.parse_args()
	Trainer.configure(args)



	train_loader = Cocodata(args).train_loader()
	val_loader = Cocodata(args).train_loader()
	print(len(train_loader))
	print(len(val_loader))
	model = build_model().cuda()

	loss = build_loss().loss(get_coco_multihead())
	optimizer = torch.optim.Adam(
			model.parameters(),
            lr=args.lr, betas=(args.momentum, args.beta2),
            weight_decay=args.weight_decay, eps=args.adam_eps, amsgrad=args.amsgrad)
	training_batches_per_epoch = len(train_loader)
	start_epoch = 0
	lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
		optimizer,
		[
				LearningRateLambda(
				[s * training_batches_per_epoch for s in args.lr_decay],
				decay_factor=args.lr_decay_factor,
				decay_epochs=args.lr_decay_epochs * training_batches_per_epoch,
				warm_up_start_epoch=args.lr_warm_up_start_epoch * training_batches_per_epoch,
				warm_up_epochs=args.lr_warm_up_epochs * training_batches_per_epoch,
				warm_up_factor=args.lr_warm_up_factor,
				warm_restart_schedule=[r * training_batches_per_epoch
				for r in args.lr_warm_restarts],
				warm_restart_duration=args.lr_warm_restart_duration * training_batches_per_epoch,
				),
		],
		last_epoch= start_epoch  * training_batches_per_epoch - 1,
		)
	checkpoint_shell = copy.deepcopy(model)
	trainer = Trainer(
		model, loss, optimizer, args.output,
		checkpoint_shell=checkpoint_shell,
		lr_scheduler=lr_scheduler,
		device=torch.device('cuda'),
		# model_meta_data={
		#     'args': vars(args),
		#     'version': __version__,
		#     'plugin_versions': plugin.versions(),
		#     'hostname': socket.gethostname(),
		# },
	)

	
	
	trainer.loop(train_loader, val_loader, start_epoch=start_epoch)