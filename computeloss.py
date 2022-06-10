import argparse
from model import get_coco_multihead
from heads import *
from lossutils import *



class CompositeLoss(torch.nn.Module):
    """Default loss since v0.13"""

    soft_clamp_value = 5.0

    def __init__(self, head_meta):
        super().__init__()
        self.n_confidences = head_meta.n_confidences
        self.n_vectors = head_meta.n_vectors
        self.n_scales = head_meta.n_scales

        self.field_names = (
            '{}.{}.c'.format(head_meta.dataset, head_meta.name),
            '{}.{}.vec'.format(head_meta.dataset, head_meta.name),
            '{}.{}.scales'.format(head_meta.dataset, head_meta.name),
        )

        self.bce_loss = BceL2()
        self.reg_loss = RegressionLoss()
        self.scale_loss = Scale()

        self.soft_clamp = None
        if self.soft_clamp_value:
            self.soft_clamp = SoftClamp(self.soft_clamp_value)

        self.weights = None
        if head_meta.training_weights is not None:
            assert len(head_meta.training_weights) == head_meta.n_fields
            self.weights = torch.Tensor(head_meta.training_weights).reshape(1, -1, 1, 1, 1)

        LOG.debug("The weights for the keypoints are %s", self.weights)
        self.bce_blackout = None
        self.previous_losses = None

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Composite Loss')
        group.add_argument('--soft-clamp', default=cls.soft_clamp_value, type=float,
                           help='soft clamp')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.soft_clamp_value = args.soft_clamp

    # pylint: disable=too-many-statements
    def forward(self, x, t):
        # LOG.debug('loss for %s', self.field_names)

        if t is None:
            return [None, None, None]
        assert x.shape[2] == 1 + self.n_confidences + self.n_vectors * 2 + self.n_scales
        assert t.shape[2] == self.n_confidences + self.n_vectors * 3 + self.n_scales
        t = torch.transpose(t, 2, 4)
        finite = torch.isfinite(t)
        t_confidence_raw = t[:, :, :, :, 0:self.n_confidences]
        bg_mask = torch.all(t_confidence_raw == 0.0, dim=4)
        c_mask = torch.all(t_confidence_raw > 0.0, dim=4)
        reg_mask = torch.all(finite[:, :, :, :, self.n_confidences:1 + self.n_vectors * 2], dim=4)
        scale_mask = torch.all(finite[:, :, :, :, self.n_confidences + self.n_vectors * 3:], dim=4)
        t_confidence_bg = t[bg_mask][:, 0:self.n_confidences]
        t_confidence = t[c_mask][:, 0:self.n_confidences]
        t_regs = t[reg_mask][:, self.n_confidences:1 + self.n_vectors * 2]
        t_sigma_min = t[reg_mask][
            :,
            self.n_confidences + self.n_vectors * 2:self.n_confidences + self.n_vectors * 3
        ]
        t_scales_reg = t[reg_mask][:, self.n_confidences + self.n_vectors * 3:]
        t_scales = t[scale_mask][:, self.n_confidences + self.n_vectors * 3:]

        x = torch.transpose(x, 2, 4)
        x_confidence_bg = x[bg_mask][:, 1:1 + self.n_confidences]
        x_logs2_c = x[c_mask][:, 0:1]
        x_confidence = x[c_mask][:, 1:1 + self.n_confidences]
        x_logs2_reg = x[reg_mask][:, 0:1]
        x_regs = x[reg_mask][:, 1 + self.n_confidences:1 + self.n_confidences + self.n_vectors * 2]
        x_scales_reg = x[reg_mask][:, 1 + self.n_confidences + self.n_vectors * 2:]
        x_scales = x[scale_mask][:, 1 + self.n_confidences + self.n_vectors * 2:]
        t_scales_reg = t_scales_reg.clone()
        invalid_t_scales_reg = torch.isnan(t_scales_reg)
        t_scales_reg[invalid_t_scales_reg] = \
            torch.nn.functional.softplus(x_scales_reg.detach()[invalid_t_scales_reg])

        l_confidence_bg = self.bce_loss(x_confidence_bg, t_confidence_bg)
        l_confidence = self.bce_loss(x_confidence, t_confidence)
        l_reg = self.reg_loss(x_regs, t_regs, t_sigma_min, t_scales_reg)
        l_scale = self.scale_loss(x_scales, t_scales)

        if self.soft_clamp is not None:
            l_confidence_bg = self.soft_clamp(l_confidence_bg)
            l_confidence = self.soft_clamp(l_confidence)
            l_reg = self.soft_clamp(l_reg)
            l_scale = self.soft_clamp(l_scale)
        x_logs2_c = 3.0 * torch.tanh(x_logs2_c / 3.0)
        l_confidence = 0.5 * l_confidence * torch.exp(-x_logs2_c) + 0.5 * x_logs2_c

        x_logs2_reg = 3.0 * torch.tanh(x_logs2_reg / 3.0)
        x_logb = 0.5 * x_logs2_reg + 0.69314
        reg_factor = torch.exp(-x_logb)
        x_logb = x_logb.unsqueeze(1)
        reg_factor = reg_factor.unsqueeze(1)
        if self.n_vectors > 1:
            x_logb = torch.repeat_interleave(x_logb, self.n_vectors, 1)
            reg_factor = torch.repeat_interleave(reg_factor, self.n_vectors, 1)
        l_reg = l_reg * reg_factor + x_logb


        if self.weights is not None:
            full_weights = torch.empty_like(t_confidence_raw)
            full_weights[:] = self.weights
            l_confidence_bg = full_weights[bg_mask] * l_confidence_bg
            l_confidence = full_weights[c_mask] * l_confidence
            l_reg = full_weights.unsqueeze(-1)[reg_mask] * l_reg
            l_scale = full_weights[scale_mask] * l_scale

        batch_size = t.shape[0]
        losses = [
            (torch.sum(l_confidence_bg) + torch.sum(l_confidence)) / batch_size,
            torch.sum(l_reg) / batch_size,
            torch.sum(l_scale) / batch_size,
        ]

        if not all(torch.isfinite(l).item() if l is not None else True for l in losses):
            raise Exception('found a loss that is not finite: {}, prev: {}'
                            ''.format(losses, self.previous_losses))
        self.previous_losses = [float(l.item()) if l is not None else None for l in losses]

        return losses

class build_loss:
    lambdas = None
    component_lambdas = None
    auto_tune_mtl = False
    auto_tune_mtl_variance = False

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('losses')
        group.add_argument('--lambdas', default=cls.lambdas, type=float, nargs='+',
                           help='prefactor for head losses by head')
        group.add_argument('--component-lambdas',
                           default=cls.component_lambdas, type=float, nargs='+',
                           help='prefactor for head losses by component')
        assert not cls.auto_tune_mtl
        group.add_argument('--auto-tune-mtl', default=False, action='store_true',
                           help=('[experimental] use Kendall\'s prescription for '
                                 'adjusting the multitask weight'))
        assert not cls.auto_tune_mtl_variance
        group.add_argument('--auto-tune-mtl-variance', default=False, action='store_true',
                           help=('[experimental] use Variance prescription for '
                                 'adjusting the multitask weight'))
        assert MultiHeadLoss.task_sparsity_weight == \
            MultiHeadLossAutoTuneKendall.task_sparsity_weight
        assert MultiHeadLoss.task_sparsity_weight == \
            MultiHeadLossAutoTuneVariance.task_sparsity_weight
        group.add_argument('--task-sparsity-weight',
                           default=MultiHeadLoss.task_sparsity_weight, type=float,
                           help='[experimental]')

        for l in set(LOSSES.values()):
            l.cli(parser)
        for lc in LOSS_COMPONENTS:
            lc.cli(parser)

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.lambdas = args.lambdas
        cls.component_lambdas = args.component_lambdas
        cls.auto_tune_mtl = args.auto_tune_mtl
        cls.auto_tune_mtl_variance = args.auto_tune_mtl_variance

        # MultiHeadLoss
        MultiHeadLoss.task_sparsity_weight = args.task_sparsity_weight
        MultiHeadLossAutoTuneKendall.task_sparsity_weight = args.task_sparsity_weight
        MultiHeadLossAutoTuneVariance.task_sparsity_weight = args.task_sparsity_weight

        for l in set(LOSSES.values()):
            l.configure(args)
        for lc in LOSS_COMPONENTS:
            lc.configure(args)

    def loss(self, head_metas):
        sparse_task_parameters = None
        

        losses = [LOSSES[meta.__class__](meta) for meta in head_metas]
        component_lambdas = self.component_lambdas
        if component_lambdas is None and self.lambdas is not None:
            assert len(self.lambdas) == len(head_metas)
            component_lambdas = [
                head_lambda
                for loss, head_lambda in zip(losses, self.lambdas)
                for _ in loss.field_names
            ]

        if self.auto_tune_mtl:
            loss = MultiHeadLossAutoTuneKendall(
                losses, component_lambdas, sparse_task_parameters=sparse_task_parameters)
        elif self.auto_tune_mtl_variance:
            loss = MultiHeadLossAutoTuneVariance(
                losses, component_lambdas, sparse_task_parameters=sparse_task_parameters)
        else:
            loss = MultiHeadLoss(losses, component_lambdas)

        return loss

LOSSES = {
    Cif: CompositeLoss,
    Caf: CompositeLoss,
}
LOSS_COMPONENTS = {
    Bce,
}

if __name__=='__main__':

    loss = build_loss().loss(get_coco_multihead())
