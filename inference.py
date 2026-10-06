"""Run batched Faster R-CNN hand/object inference on a video."""

import argparse
import os

import cv2
import numpy as np
import torch

import _init_paths  # noqa: F401
from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.vgg16 import vgg16
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.roi_layers import nms
from model.utils.blob import im_list_to_blob
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.utils.net_utils import vis_detections_filtered_objects_PIL


CLASSES = ('__background__', 'targetobject', 'hand')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, help='Video path or camera index')
    parser.add_argument('--output', default='video_det.mp4', help='Annotated output video')
    parser.add_argument('--load_dir', default='models')
    parser.add_argument('--dataset', default='pascal_voc')
    parser.add_argument('--net', choices=('vgg16', 'res50', 'res101', 'res152'), default='res101')
    parser.add_argument('--cfg', dest='cfg_file', default='cfgs/res101.yml')
    parser.add_argument('--set', dest='set_cfgs', nargs=argparse.REMAINDER)
    parser.add_argument('--checksession', type=int, default=1)
    parser.add_argument('--checkepoch', type=int, default=8)
    parser.add_argument('--checkpoint', type=int, required=True)
    parser.add_argument('--batch_size', '--bs', type=int, default=4)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--class_agnostic', action='store_true')
    parser.add_argument('--thresh_hand', type=float, default=0.5)
    parser.add_argument('--thresh_obj', type=float, default=0.5)
    parser.add_argument('--no_vis', action='store_true', help='Skip writing annotated output')
    parser.add_argument('--print_detections', action='store_true')
    return parser.parse_args()


def video_to_batch_images(capture, batch_size):
    """Yield groups of (frame indices, BGR frames) from an open capture."""
    if batch_size < 1:
        raise ValueError('batch_size must be at least 1')
    frame_index = 0
    while True:
        frames, indices = [], []
        for _ in range(batch_size):
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
            indices.append(frame_index)
            frame_index += 1
        if not frames:
            return
        yield indices, frames


def prepare_batch(frames):
    """Normalize and resize frames, returning an NCHW blob and per-frame metadata."""
    processed, metadata, scales = [], [], []
    for frame in frames:
        image = frame.astype(np.float32, copy=True) - cfg.PIXEL_MEANS
        height, width = image.shape[:2]
        scale = float(cfg.TEST.SCALES[0]) / min(height, width)
        if np.round(scale * max(height, width)) > cfg.TEST.MAX_SIZE:
            scale = float(cfg.TEST.MAX_SIZE) / max(height, width)
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        processed.append(resized)
        metadata.append((resized.shape[0], resized.shape[1], scale))
        scales.append(scale)
    blob = im_list_to_blob(processed)
    return torch.from_numpy(blob).permute(0, 3, 1, 2), torch.tensor(metadata), scales


def decode_batch(rois, cls_prob, bbox_pred, loss_list, im_info, scales, args):
    """Decode and NMS detections independently for each image in a batched result."""
    scores = cls_prob
    boxes = rois[:, :, 1:5]
    if cfg.TEST.BBOX_REG:
        deltas = bbox_pred
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            stds = deltas.new_tensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
            means = deltas.new_tensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
            deltas = (deltas.reshape(-1, 4) * stds + means).view_as(bbox_pred)
        pred_boxes = clip_boxes(bbox_transform_inv(boxes, deltas, boxes.size(0)), im_info, boxes.size(0))
    else:
        pred_boxes = boxes.repeat(1, 1, scores.size(2))

    # The extension layer returns flattened ROI predictions as [1, B*R, C].
    batch_size, num_rois = scores.shape[:2]
    contact_scores = loss_list[0][0].reshape(batch_size, num_rois, -1)
    contact = contact_scores.argmax(dim=2).float()
    offsets = loss_list[1][0].reshape(batch_size, num_rois, -1).detach()
    side_logits = loss_list[2][0].reshape(batch_size, num_rois, -1).detach()
    left_right = (torch.sigmoid(side_logits) > 0.5).float()
    results = []
    for batch_index, scale in enumerate(scales):
        image_scores = scores[batch_index]
        image_boxes = pred_boxes[batch_index] / scale
        detections = {}
        for class_index, key, threshold in ((1, 'obj_dets', args.thresh_obj),
                                             (2, 'hand_dets', args.thresh_hand)):
            inds = torch.nonzero(image_scores[:, class_index] > threshold, as_tuple=False).view(-1)
            if not inds.numel():
                detections[key] = None
                continue
            class_boxes = image_boxes[inds] if args.class_agnostic else image_boxes[inds, class_index * 4:(class_index + 1) * 4]
            class_scores = image_scores[inds, class_index]
            order = class_scores.argsort(descending=True)
            rows = torch.cat((class_boxes, class_scores[:, None], contact[batch_index, inds, None],
                              offsets[batch_index, inds], left_right[batch_index, inds]), dim=1)[order]
            keep = nms(class_boxes[order], class_scores[order], cfg.TEST.NMS).long()
            detections[key] = rows[keep].cpu().numpy()
        results.append(detections)
    return results


def build_model(args, device):
    model_dir = os.path.join(args.load_dir, '{}_handobj_100K'.format(args.net), args.dataset)
    checkpoint_path = os.path.join(model_dir, 'faster_rcnn_{}_{}_{}.pth'.format(
        args.checksession, args.checkepoch, args.checkpoint))
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError('Checkpoint not found: {}'.format(checkpoint_path))
    if args.net == 'vgg16':
        model = vgg16(CLASSES, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        depth = {'res50': 50, 'res101': 101, 'res152': 152}[args.net]
        model = resnet(CLASSES, depth, pretrained=False, class_agnostic=args.class_agnostic)
    model.create_architecture()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint:
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    # This repository's ResNet overrides train() without returning self, so
    # nn.Module.eval() would return None. Set eval mode directly and keep model.
    model = model.to(device)
    model.train(False)
    return model


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError('--batch_size must be at least 1')
    cfg_from_file(args.cfg_file)
    if args.set_cfgs:
        cfg_from_list(args.set_cfgs)
    cfg_from_list(['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5, 1, 2]'])
    cfg.USE_GPU_NMS = args.cuda
    cfg.CUDA = args.cuda
    device = torch.device('cuda' if args.cuda else 'cpu')
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    model = build_model(args, device)

    source = int(args.video) if args.video.isdigit() else args.video
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError('Could not open video source: {}'.format(args.video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    frame_count = 0
    try:
        with torch.no_grad():
            for indices, frames in video_to_batch_images(capture, args.batch_size):
                im_data, im_info, scales = prepare_batch(frames)
                im_data = im_data.to(device)
                im_info = im_info.to(device)
                batch_count = len(frames)
                gt_boxes = torch.zeros((batch_count, 1, 5), device=device)
                num_boxes = torch.zeros((batch_count,), dtype=torch.long, device=device)
                box_info = torch.zeros((batch_count, 1, 5), device=device)
                outputs = model(im_data, im_info, gt_boxes, num_boxes, box_info)
                results = decode_batch(outputs[0], outputs[1], outputs[2], outputs[8],
                                       im_info, scales, args)
                for index, frame, detections in zip(indices, frames, results):
                    if args.print_detections:
                        print('Frame {}: {}'.format(index, detections))
                    if not args.no_vis:
                        rendered = vis_detections_filtered_objects_PIL(
                            frame,
                            detections['obj_dets'], detections['hand_dets'],
                            args.thresh_hand, args.thresh_obj)
                        frame = cv2.cvtColor(np.asarray(rendered.convert('RGB')), cv2.COLOR_RGB2BGR)
                        if writer is None:
                            height, width = frame.shape[:2]
                            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
                            writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'),
                                                     fps, (width, height))
                            if not writer.isOpened():
                                raise RuntimeError('Could not create output video: {}'.format(args.output))
                        writer.write(frame)
                    frame_count += 1
                print('Processed {} frames'.format(frame_count), end='\r', flush=True)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    print('\nFinished: {} frames'.format(frame_count))
    if not args.no_vis:
        print('Annotated video: {}'.format(args.output))


if __name__ == '__main__':
    main()
