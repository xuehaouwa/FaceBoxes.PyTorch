import torch
from gv_tools.tracking.tracking_region import TrackingRegion
from models.faceboxes import FaceBoxes
from gv_tools.util.logger import Logger
from typing import List
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode
from utils.nms_wrapper import nms
import numpy as np
from data import cfg


class FaceBoxDetector:
    def __init__(self, min_score=0.9, use_gpu=True):
        # Minimum score to consider as a detection.
        self.score_min = min_score

        self.net = FaceBoxes(phase='test', size=None, num_classes=2)

        self.use_gpu = use_gpu

        self.logger = Logger()

    def load_model(self, path_to_model):
        self.logger.field('Loading pretrained model from', path_to_model)
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(path_to_model, map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        self.check_keys(self.net, pretrained_dict)
        self.net.load_state_dict(pretrained_dict, strict=False)
        self.net.eval()

        if self.use_gpu:
            self.net.cuda()

    def detect(self, images) -> List[List[TrackingRegion]]:
        frames = []
        for img in images:
            img = np.float32(img)
            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            if self.use_gpu:
                img = img.cuda()
                scale = scale.cuda()
            out = self.net(img)
            face_regions = self.nms_process(out, scale, im_height, im_width)
            frames.append(face_regions)

        return frames

    def nms_process(self, network_output, scale, im_height, im_width) -> List[TrackingRegion]:
        priorbox = PriorBox(cfg, network_output[2], (im_height, im_width), phase='test')
        priors = priorbox.forward()
        if self.use_gpu:
            priors = priors.cuda()
        loc, conf, _ = network_output
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > self.score_min)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS, top_k = 5
        order = scores.argsort()[::-1][:5000]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(dets, 0.3, force_cpu=False)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:750, :]

        regions = []

        for i in range(dets.shape[0]):
            face_region = TrackingRegion()
            face_region.set_rect(left=dets[i, 0], top=dets[i, 1], right=dets[i, 2], bottom=dets[i, 3])
            face_region.confidence = dets[i, 4]
            face_region.data["class_id"] = "face"
            regions.append(face_region)

        return regions

    def check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        self.logger.field('Missing keys', len(missing_keys))
        self.logger.field('Unused checkpoint keys', len(unused_pretrained_keys))
        self.logger.field('Used keys', len(used_pretrained_keys))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True

    @staticmethod
    def remove_prefix(state_dict, prefix):
        """ Old style model is stored with all names of parameters sharing common prefix 'module.' """
        print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}




