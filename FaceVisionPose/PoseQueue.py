import torch
import torchvision.transforms as transforms
import mobilenet_v1
import numpy as np
import cv2
import dlib
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors
from utils.cv_plot import plot_pose_box
from utils.estimate_pose import parse_pose
from utils.render import get_depths_image, cget_depths_image, cpncc
from utils.paf import gen_img_paf
import argparse
import torch.backends.cudnn as cudnn
import paho.mqtt.client as mqtt

STD_SIZE = 120

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe("filename")

def on_message(client, userdata, msg):
	image = msg.payload.decode()
	run(image)
    # client.disconnect()

def run(image):
    # 1. load pre-tained model
    checkpoint_fp = 'models/phase1_wpdc_vdc.pth.tar'
    arch = 'mobilenet_1'

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    model.eval()

    # 2. load dlib model for face detection and landmark used for face cropping
    dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
    face_regressor = dlib.shape_predictor(dlib_landmark_model)
    face_detector = dlib.get_frontal_face_detector()

    # 3. forward
    tri = sio.loadmat('tri.mat')['tri']
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

    img_ori = cv2.imread(image)
    # img_ori = img_ori[:, :, [2, 1, 0]]
    rects = face_detector(img_ori, 1)

    # if len(rects) == 0:
    #     rects = dlib.rectangles()
    #     rect_fp = img_fp + '.bbox'
    #     lines = open(rect_fp).read().strip().split('\n')[1:]
    #     for l in lines:
    #         l, r, t, b = [int(_) for _ in l.split(' ')[1:]]
    #         rect = dlib.rectangle(l, r, t, b)
    #         rects.append(rect)

    pts_res = []
    Ps = []  # Camera matrix collection
    poses = []  # pose collection, [todo: validate it]
    vertices_lst = []  # store multiple face vertices
    ind = 0
    suffix = get_suffix(image)
    for rect in rects:
        # whether use dlib landmark to crop image, if not, use only face bbox to calc roi bbox for cropping
        # - use landmark for cropping
        pts = face_regressor(img_ori, rect).parts()
        pts = np.array([[pt.x, pt.y] for pt in pts]).T
        roi_box = parse_roi_box_from_landmark(pts)

        img = crop_img(img_ori, roi_box)

        # forward: one step
        img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
        input = transform(img).unsqueeze(0)
        with torch.no_grad():
            param = model(input)
            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

        # 68 pts
        pts68 = predict_68pts(param, roi_box)

        # two-step for more accurate bbox to crop face

        pts_res.append(pts68)
        P, pose = parse_pose(param)
        Ps.append(P)
        poses.append(pose)

        vertices = predict_dense(param, roi_box)
        vertices_lst.append(vertices)

        # dense face 3d vertices
        wfp = './output/obj_{}.obj'.format(image.split('/')[-1])
        colors = get_colors(img_ori, vertices)
        write_obj_with_colors(wfp, vertices, tri, colors)
        print('Dump obj with sampled texture to {}'.format(wfp))
        ind += 1

if __name__ == "__main__":
    client = mqtt.Client()
    client.connect("localhost")

    client.on_connect = on_connect
    client.on_message = on_message

    client.loop_forever()
