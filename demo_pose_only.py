# coding: utf-8
# Pose-only demo: pure ONNX — no .yml, no .pkl, no BFM.
# Loads two ONNX files (face detector + pose-baked TDDFA) and decodes Euler angles.

import argparse
import glob
import os
import os.path as osp
import sys
from math import asin, atan2, cos, pi

import cv2
import numpy as np
import onnxruntime

from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX


def parse_roi_box_from_bbox(bbox):
    left, top, right, bottom = bbox[:4]
    old_size = (right - left + bottom - top) / 2
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0 + old_size * 0.14
    size = int(old_size * 1.58)
    sx, sy = center_x - size / 2, center_y - size / 2
    return [sx, sy, sx + size, sy + size]


def crop_img(img, roi_box):
    h, w = img.shape[:2]
    sx, sy, ex, ey = (int(round(v)) for v in roi_box)
    dh, dw = ey - sy, ex - sx
    out = np.zeros((dh, dw, 3), dtype=np.uint8) if img.ndim == 3 else np.zeros((dh, dw), dtype=np.uint8)
    sx_, ex_ = max(0, sx), min(w, ex)
    sy_, ey_ = max(0, sy), min(h, ey)
    out[sy_ - sy:ey_ - sy, sx_ - sx:ex_ - sx] = img[sy_:ey_, sx_:ex_]
    return out


def decode_pose(param):
    """param: (62,) float32 — already denormalized inside the ONNX.
    Returns (yaw, pitch, roll) in degrees and the 3x4 affine P (for axis drawing)."""
    P = param[:12].reshape(3, 4)
    R1, R2 = P[0:1, :3], P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)
    R = np.concatenate((r1, r2, r3), 0)
    if R[2, 0] > 0.998:
        z, x = 0.0, pi / 2; y = z + atan2(-R[0, 1], -R[0, 2])
    elif R[2, 0] < -0.998:
        z, x = 0.0, -pi / 2; y = -z + atan2(R[0, 1], R[0, 2])
    else:
        x = asin(R[2, 0])
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))
    yaw, pitch, roll = (a * 180 / pi for a in (x, y, z))
    return yaw, pitch, roll, R, s


def draw_pose(img, bbox, yaw, pitch, roll, R, s, color=(40, 255, 0), thickness=1):
    """Wireframe 3D box overlay matching the original 3DDFA_V2 viz_pose look,
    but sized from the face bbox instead of the sparse 68-vertex hypotenuse."""
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    side = max(x2 - x1, y2 - y1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2 + side * 0.05   # small downshift toward jaw, less than the ROI 0.14 used for cropping
    # The original calc_hypotenuse runs on the sparse 68-vertex bbox, which is
    # tighter than the FaceBoxes detector bbox (no forehead/ears). Compensate
    # by a 0.85 shrink so the rear square spans roughly 0.8 * side.
    rear = np.sqrt(2) * side / 3 * 0.85
    front = rear * 4 / 3

    # 10-point wireframe: rear square (closed) + front square (closed)
    p3d = np.float32([
        [-rear, -rear, 0], [-rear,  rear, 0], [ rear,  rear, 0], [ rear, -rear, 0], [-rear, -rear, 0],
        [-front, -front, front], [-front,  front, front], [ front,  front, front], [ front, -front, front], [-front, -front, front],
    ])

    p2d = (R @ p3d.T).T[:, :2]
    p2d[:, 1] *= -1  # image y axis is flipped
    p2d -= p2d[:4].mean(axis=0)  # centre back-face on origin
    p2d[:, 0] += cx
    p2d[:, 1] += cy
    p2d = p2d.astype(np.int32)

    cv2.polylines(img, [p2d], True, color, thickness, cv2.LINE_AA)
    for a, b in ((1, 6), (2, 7), (3, 8)):
        cv2.line(img, tuple(p2d[a]), tuple(p2d[b]), color, thickness, cv2.LINE_AA)

    # Axis gizmo (X red, Y green, Z blue) — same projection convention as the box.
    axes_3d = np.float32([[rear, 0, 0], [0, rear, 0], [0, 0, rear]])
    axes_2d = (R @ axes_3d.T)[:2].T
    axes_2d[:, 1] *= -1
    origin = (int(cx), int(cy))
    for v, c in zip(axes_2d, [(0, 0, 200), (0, 200, 0), (200, 0, 0)]):
        cv2.arrowedLine(img, origin,
                        (int(origin[0] + v[0]), int(origin[1] + v[1])),
                        c, 2, tipLength=0.15)

    cv2.putText(img, f'y={yaw:+.0f} p={pitch:+.0f} r={roll:+.0f}',
                (int(x1), max(0, int(y1) - 6)),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


SIZE = 120  # all baked pose ONNX checkpoints take 120x120


def main(args):
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'True')
    os.environ.setdefault('OMP_NUM_THREADS', '4')

    sess_opts = onnxruntime.SessionOptions()
    sess_opts.intra_op_num_threads = int(os.environ['OMP_NUM_THREADS'])
    pose_sess = onnxruntime.InferenceSession(args.model, sess_opts)
    face_boxes = FaceBoxes_ONNX()

    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    img_fps = sorted(p for ext in exts for p in glob.glob(osp.join(args.input_dir, ext)))
    if not img_fps:
        print(f'No images found in {args.input_dir}'); sys.exit(-1)
    os.makedirs(args.output_dir, exist_ok=True)

    for img_fp in img_fps:
        img = cv2.imread(img_fp)
        bboxes = face_boxes(img)
        n = len(bboxes)
        print(f'\n[{img_fp}] detect {n} face(s)')
        if n == 0:
            continue

        stem = osp.splitext(osp.basename(img_fp))[0]
        for i, bbox in enumerate(bboxes):
            roi = parse_roi_box_from_bbox(bbox)
            crop = cv2.resize(crop_img(img, roi), (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
            crop_fp = osp.join(args.output_dir, f'{stem}_crop{i}.png')
            cv2.imwrite(crop_fp, crop)
            print(f'  crop -> {crop_fp}')
            inp = ((crop.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...] - 127.5) / 128.)
            param = pose_sess.run(None, {'input': inp})[0].flatten()
            yaw, pitch, roll, R, s = decode_pose(param)
            print(f'  yaw={yaw:+6.1f}  pitch={pitch:+6.1f}  roll={roll:+6.1f}')
            draw_pose(img, bbox, yaw, pitch, roll, R, s)

        wfp = osp.join(args.output_dir, stem + '_pose.png')
        cv2.imwrite(wfp, img)
        print(f'  -> {wfp}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose-only demo, pure ONNX (no yml/pkl)')
    parser.add_argument('-m', '--model', type=str, default='weights/mb1_120x120_pose.onnx',
                        help='Pose ONNX with mean/std baked in')
    parser.add_argument('-i', '--input_dir', type=str, default='examples/inputs')
    parser.add_argument('-d', '--output_dir', type=str, default='examples/results')
    main(parser.parse_args())
