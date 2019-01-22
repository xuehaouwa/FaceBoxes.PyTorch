import torch
import cv2
import numpy as np
from data import cfg
from models.faceboxes import FaceBoxes
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode
from utils.nms_wrapper import nms

trained_model_path = 'weights/FaceBoxes.pth'


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path):
    print('Loading pretrained model from {}'.format(pretrained_path))
    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


model = FaceBoxes(phase='test', size=None, num_classes=2)
device = torch.device("cuda")
model = load_model(model, trained_model_path)
model = model.to(device)
model.eval()

image_path = '/home/haoxue/Pictures/face.jpg'
img = np.float32(cv2.imread(image_path, cv2.IMREAD_COLOR))
im_height, im_width, _ = img.shape

scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
img -= (104, 117, 123)
img = img.transpose(2, 0, 1)
img = torch.from_numpy(img).unsqueeze(0)
# if args.cuda:
img = img.cuda()
scale = scale.cuda()
out = model(img)
priorbox = PriorBox(cfg, out[2], (im_height, im_width), phase='test')
priors = priorbox.forward()
priors = priors.cuda()
loc, conf, _ = out
prior_data = priors.data
boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
boxes = boxes * scale
boxes = boxes.cpu().numpy()
scores = conf.data.cpu().numpy()[:, 1]

# ignore low scores
confidence_threshold = 0.9
inds = np.where(scores > confidence_threshold)[0]
boxes = boxes[inds]
scores = scores[inds]
# print('boxes', boxes)

# keep top-K before NMS
order = scores.argsort()[::-1][:5]
boxes = boxes[order]
scores = scores[order]

# do NMS
dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
keep = nms(dets, 0.3, force_cpu=False)
dets = dets[keep, :]

# keep top-K faster NMS
dets = dets[:750, :]

# print('boxes_list',boxes_list)
# dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
# print('dets', dets)
# dets = dets[:5, :]
# print('dets[:5, :]', dets)
for k in range(dets.shape[0]):
    xmin = dets[k, 0]
    ymin = dets[k, 1]
    xmax = dets[k, 2]
    ymax = dets[k, 3]
    ymin += 0.2 * (ymax - ymin + 1)
    score = dets[k, 4]
    print('xmin {:.1f} ymin {:.1f} xmax {:.1f} ymax {:.1f}\n'.format(xmin, ymin, xmax, ymax))
