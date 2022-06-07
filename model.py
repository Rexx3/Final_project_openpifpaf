import torch
import shufflenet
import torchvision.models as models
from heads import *


'''
constants
'''
COCO_PERSON_SKELETON = [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13),
    (6, 7), (6, 8), (7, 9), (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
    (2, 4), (3, 5), (4, 6), (5, 7),
]


KINEMATIC_TREE_SKELETON = [
    (1, 2), (2, 4),  # left head
    (1, 3), (3, 5),
    (1, 6),
    (6, 8), (8, 10),  # left arm
    (1, 7),
    (7, 9), (9, 11),  # right arm
    (6, 12), (12, 14), (14, 16),  # left side
    (7, 13), (13, 15), (15, 17),
]


COCO_KEYPOINTS = [
    'nose',            # 1
    'left_eye',        # 2
    'right_eye',       # 3
    'left_ear',        # 4
    'right_ear',       # 5
    'left_shoulder',   # 6
    'right_shoulder',  # 7
    'left_elbow',      # 8
    'right_elbow',     # 9
    'left_wrist',      # 10
    'right_wrist',     # 11
    'left_hip',        # 12
    'right_hip',       # 13
    'left_knee',       # 14
    'right_knee',      # 15
    'left_ankle',      # 16
    'right_ankle',     # 17
]


COCO_UPRIGHT_POSE = np.array([
    [0.0, 9.3, 2.0],  # 'nose',            # 1
    [-0.35, 9.7, 2.0],  # 'left_eye',        # 2
    [0.35, 9.7, 2.0],  # 'right_eye',       # 3
    [-0.7, 9.5, 2.0],  # 'left_ear',        # 4
    [0.7, 9.5, 2.0],  # 'right_ear',       # 5
    [-1.4, 8.0, 2.0],  # 'left_shoulder',   # 6
    [1.4, 8.0, 2.0],  # 'right_shoulder',  # 7
    [-1.75, 6.0, 2.0],  # 'left_elbow',      # 8
    [1.75, 6.2, 2.0],  # 'right_elbow',     # 9
    [-1.75, 4.0, 2.0],  # 'left_wrist',      # 10
    [1.75, 4.2, 2.0],  # 'right_wrist',     # 11
    [-1.26, 4.0, 2.0],  # 'left_hip',        # 12
    [1.26, 4.0, 2.0],  # 'right_hip',       # 13
    [-1.4, 2.0, 2.0],  # 'left_knee',       # 14
    [1.4, 2.1, 2.0],  # 'right_knee',      # 15
    [-1.4, 0.0, 2.0],  # 'left_ankle',      # 16
    [1.4, 0.1, 2.0],  # 'right_ankle',     # 17
])


COCO_DAVINCI_POSE = np.array([
    [0.0, 9.3, 2.0],  # 'nose',            # 1
    [-0.35, 9.7, 2.0],  # 'left_eye',        # 2
    [0.35, 9.7, 2.0],  # 'right_eye',       # 3
    [-0.7, 9.5, 2.0],  # 'left_ear',        # 4
    [0.7, 9.5, 2.0],  # 'right_ear',       # 5
    [-1.4, 8.0, 2.0],  # 'left_shoulder',   # 6
    [1.4, 8.0, 2.0],  # 'right_shoulder',  # 7
    [-3.3, 9.0, 2.0],  # 'left_elbow',      # 8
    [3.3, 9.2, 2.0],  # 'right_elbow',     # 9
    [-4.5, 10.5, 2.0],  # 'left_wrist',      # 10
    [4.5, 10.7, 2.0],  # 'right_wrist',     # 11
    [-1.26, 4.0, 2.0],  # 'left_hip',        # 12
    [1.26, 4.0, 2.0],  # 'right_hip',       # 13
    [-2.0, 2.0, 2.0],  # 'left_knee',       # 14
    [2.0, 2.1, 2.0],  # 'right_knee',      # 15
    [-2.4, 0.0, 2.0],  # 'left_ankle',      # 16
    [2.4, 0.1, 2.0],  # 'right_ankle',     # 17
])


HFLIP = {
    'left_eye': 'right_eye',
    'right_eye': 'left_eye',
    'left_ear': 'right_ear',
    'right_ear': 'left_ear',
    'left_shoulder': 'right_shoulder',
    'right_shoulder': 'left_shoulder',
    'left_elbow': 'right_elbow',
    'right_elbow': 'left_elbow',
    'left_wrist': 'right_wrist',
    'right_wrist': 'left_wrist',
    'left_hip': 'right_hip',
    'right_hip': 'left_hip',
    'left_knee': 'right_knee',
    'right_knee': 'left_knee',
    'left_ankle': 'right_ankle',
    'right_ankle': 'left_ankle',
}


DENSER_COCO_PERSON_SKELETON = [
    (1, 2), (1, 3), (2, 3), (1, 4), (1, 5), (4, 5),
    (1, 6), (1, 7), (2, 6), (3, 7),
    (2, 4), (3, 5), (4, 6), (5, 7), (6, 7),
    (6, 12), (7, 13), (6, 13), (7, 12), (12, 13),
    (6, 8), (7, 9), (8, 10), (9, 11), (6, 10), (7, 11),
    (8, 9), (10, 11),
    (10, 12), (11, 13),
    (10, 14), (11, 15),
    (14, 12), (15, 13), (12, 15), (13, 14),
    (12, 16), (13, 17),
    (16, 14), (17, 15), (14, 17), (15, 16),
    (14, 15), (16, 17),
]


DENSER_COCO_PERSON_CONNECTIONS = [
    c
    for c in DENSER_COCO_PERSON_SKELETON
    if c not in COCO_PERSON_SKELETON
]


COCO_PERSON_SIGMAS = [
    0.026,  # nose
    0.025,  # eyes
    0.025,  # eyes
    0.035,  # ears
    0.035,  # ears
    0.079,  # shoulders
    0.079,  # shoulders
    0.072,  # elbows
    0.072,  # elbows
    0.062,  # wrists
    0.062,  # wrists
    0.107,  # hips
    0.107,  # hips
    0.087,  # knees
    0.087,  # knees
    0.089,  # ankles
    0.089,  # ankles
]
HEADS = {
    Cif: CompositeField4,
    Caf: CompositeField4,
    CifDet: CompositeField4,
}
COCO_PERSON_SCORE_WEIGHTS = [3.0] * 3 + [1.0] * (len(COCO_KEYPOINTS) - 3)
'''
'''


def get_coco_multihead():

	cif = Cif('cif', 'cocokp',	keypoints=COCO_KEYPOINTS,
                                      sigmas=COCO_PERSON_SIGMAS,
                                      pose=COCO_UPRIGHT_POSE,
                                      draw_skeleton=COCO_PERSON_SKELETON,
                                      score_weights=COCO_PERSON_SCORE_WEIGHTS)
	caf = Caf('caf', 'cocokp', keypoints=COCO_KEYPOINTS, sigmas=COCO_PERSON_SIGMAS, pose=COCO_UPRIGHT_POSE, skeleton=COCO_PERSON_SKELETON)
	dcaf = Caf('caf25', 'cocokp',
                                   keypoints=COCO_KEYPOINTS,
                                   sigmas=COCO_PERSON_SIGMAS,
                                   pose=COCO_UPRIGHT_POSE,
                                   skeleton=DENSER_COCO_PERSON_CONNECTIONS,
                                   sparse_skeleton=COCO_PERSON_SKELETON,
                                   only_in_field_of_view=True)
	cif.base_stride = 16
	caf.base_stride = 16
	cif.upsample_stride = 2
	caf.upsample_stride = 2
	dcaf.upsample_stride = 1
	head_metas = [cif, caf]
	return head_metas


class Openpifpaf(torch.nn.Module):
    def __init__(self, base_net, head_nets, *,
                 process_input=None, process_heads=None):
        super().__init__()

        self.base_net = base_net
        self.head_nets = None
        self.process_input = process_input
        self.process_heads = process_heads

        self.set_head_nets(head_nets)

    @property
    def head_metas(self):
        if self.head_nets is None:
            return None
        return [hn.meta for hn in self.head_nets]

    def set_head_nets(self, head_nets):
        if not isinstance(head_nets, torch.nn.ModuleList):
            head_nets = torch.nn.ModuleList(head_nets)

        for hn_i, hn in enumerate(head_nets):
            hn.meta.head_index = hn_i
            hn.meta.base_stride = self.base_net.stride

        self.head_nets = head_nets

    def forward(self, image_batch, *, head_mask=None):
        if self.process_input is not None:
            image_batch = self.process_input(image_batch)

        x = self.base_net(image_batch)
        if head_mask is not None:
            head_outputs = tuple(hn(x) if m else None for hn, m in zip(self.head_nets, head_mask))
        else:
            head_outputs = tuple(hn(x) for hn in self.head_nets)

        if self.process_heads is not None:
            head_outputs = self.process_heads(head_outputs)

        return head_outputs

def build_model():
	backbone = shufflenet.ShuffleNetV2K('shufflenetv2k16', [4, 8, 4], [24, 348, 696, 1392, 1392])
	headnets = [HEADS[h.__class__](h, backbone.out_features) for h in get_coco_multihead()]
	net = Openpifpaf(backbone, headnets)
	return net

if __name__=='__main__':

	print(net)


