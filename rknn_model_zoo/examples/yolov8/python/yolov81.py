import os
import cv2
import sys
import argparse

# add path
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('rknn_model_zoo')+1]))

from py_utils.coco_utils import COCO_test_helper
import numpy as np


OBJ_THRESH = 0.25
NMS_THRESH = 0.45

# The follew two param is for map test
# OBJ_THRESH = 0.001
# NMS_THRESH = 0.65

IMG_SIZE = (640, 640)  # (width, height), such as (1280, 736)

CLASSES = ("general_corrosion","pitting_corrosion","uniform_corrosion","crevice_corrosion")

coco_id_list = [4]


def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score* box_confidences >= OBJ_THRESH)
    scores = (class_max_score* box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

def dfl(position):
    # Distribution Focal Loss (DFL)
    import torch
    x = torch.tensor(position)
    n,c,h,w = x.shape
    p_num = 4
    mc = c//p_num
    y = x.reshape(n,p_num,mc,h,w)
    y = y.softmax(2)
    acc_metrix = torch.tensor(range(mc)).float().reshape(1,1,mc,1,1)
    y = (y*acc_metrix).sum(2)
    return y.numpy()

def xywh2xyxy(boxes):
    """Convert center-x/center-y/width/height boxes to x1/y1/x2/y2."""
    boxes = boxes.copy()
    cx = boxes[:, 0].copy()
    cy = boxes[:, 1].copy()
    w = boxes[:, 2].copy()
    h = boxes[:, 3].copy()
    boxes[:, 0] = cx - w / 2
    boxes[:, 1] = cy - h / 2
    boxes[:, 2] = cx + w / 2
    boxes[:, 3] = cy + h / 2
    return boxes

def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

    position = dfl(position)
    box_xy  = grid +0.5 -position[:,0:2,:,:]
    box_xy2 = grid +0.5 +position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

    return xyxy

def post_process_single_output(output):
    """Post-process original ONNX export with one detection tensor.

    Expected shape is [1, C, N] or [1, N, C], where the first 4 channels are
    cx, cy, w, h and the remaining channels are class scores.
    """
    output = np.asarray(output)
    if output.ndim != 3:
        raise ValueError("single-output model expects a 3D tensor, got {}".format(output.shape))

    # Normalize to [N, C].
    if output.shape[1] <= output.shape[2]:
        pred = output[0].transpose(1, 0)
    else:
        pred = output[0]

    if pred.shape[1] < 5:
        raise ValueError("single-output tensor must contain 4 box values and class scores, got {}".format(pred.shape))

    boxes = pred[:, :4].astype(np.float32)
    classes_conf = pred[:, 4:].astype(np.float32)
    boxes = xywh2xyxy(boxes)

    # Match the existing filter/nms path: objectness is absent, so use 1.0.
    box_confidences = np.ones((boxes.shape[0], 1), dtype=np.float32)
    boxes, classes, scores = filter_boxes(boxes, box_confidences, classes_conf)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores

def post_process(input_data):
    if isinstance(input_data, np.ndarray):
        input_data = [input_data]

    # 新增：如果是单个 3D 张量，走单输出后处理
    if len(input_data) == 1 and np.asarray(input_data[0]).ndim == 3:
        return post_process_single_output(input_data[0])

    # 以下是原三分支处理逻辑（保持不变）
    boxes, scores, classes_conf = [], [], []
    defualt_branch=3
    pair_per_branch = len(input_data)//defualt_branch
    # Python 忽略 score_sum 输出
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
        classes_conf.append(input_data[pair_per_branch*i+1])
        scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # nms
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def setup_model(args):
    model_path = args.model_path
    if model_path.endswith('.pt') or model_path.endswith('.torchscript'):
        platform = 'pytorch'
        from py_utils.pytorch_executor import Torch_model_container
        model = Torch_model_container(args.model_path)
    elif model_path.endswith('.rknn'):
        platform = 'rknn'
        from py_utils.rknn_executor import RKNN_model_container 
        model = RKNN_model_container(args.model_path, args.target, args.device_id)
    elif model_path.endswith('onnx'):
        platform = 'onnx'
        from py_utils.onnx_executor import ONNX_model_container
        model = ONNX_model_container(args.model_path)
    else:
        assert False, "{} is not rknn/pytorch/onnx model".format(model_path)
    print('Model-{} is {} model, starting val'.format(model_path, platform))
    return model, platform

def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    # basic params
    parser.add_argument('--model_path', type=str, required= True, help='model path, could be .pt or .rknn file')
    parser.add_argument('--target', type=str, default=None, help='target RKNPU platform')
    parser.add_argument('--device_id', type=str, default=None, help='device id')
    
    parser.add_argument('--img_show', action='store_true', default=False, help='draw the result and show')
    parser.add_argument('--img_save', action='store_true', default='../result', help='save the result')

    # data params
    parser.add_argument('--anno_json', type=str, default='../../../datasets/COCO/annotations/instances_val2017.json', help='coco annotation path')
    # coco val folder: '../../../datasets/COCO//val2017'
    parser.add_argument('--img_folder', type=str, default='../test', help='img folder path')
    parser.add_argument('--coco_map_test', action='store_true', help='enable coco map test')

    args = parser.parse_args()

    # init model
    model, platform = setup_model(args)

    file_list = sorted(os.listdir(args.img_folder))
    img_list = []
    for path in file_list:
        if img_check(path):
            img_list.append(path)
    co_helper = COCO_test_helper(enable_letter_box=True)

    # run test
    for i in range(len(img_list)):
        print('infer {}/{}'.format(i+1, len(img_list)), end='\r')

        img_name = img_list[i]
        img_path = os.path.join(args.img_folder, img_name)
        if not os.path.exists(img_path):
            print("{} is not found", img_name)
            continue

        img_src = cv2.imread(img_path)
        if img_src is None:
            continue

        '''
        # using for test input dumped by C.demo
        img_src = np.fromfile('./input_b/demo_c_input_hwc_rgb.txt', dtype=np.uint8).reshape(640,640,3)
        img_src = cv2.cvtColor(img_src, cv2.COLOR_RGB2BGR)
        '''

        # Due to rga init with (0,0,0), we using pad_color (0,0,0) instead of (114, 114, 114)
        pad_color = (0,0,0)
        img = co_helper.letter_box(im= img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # preprocee if not rknn model
        if platform in ['pytorch', 'onnx']:
            input_data = img.transpose((2,0,1))
            input_data = input_data.reshape(1,*input_data.shape).astype(np.float32)
            input_data = input_data/255.
        else:
            input_data = img

        outputs = model.run([input_data])
        boxes, classes, scores = post_process(outputs)

        if args.img_show or args.img_save:
            print('\n\nIMG: {}'.format(img_name))
            img_p = img_src.copy()
            if boxes is not None:
                draw(img_p, co_helper.get_real_box(boxes), scores, classes)

            if args.img_save:
                if not os.path.exists('./result'):
                    os.mkdir('./result')
                result_path = os.path.join('./result', img_name)
                cv2.imwrite(result_path, img_p)
                print('Detection result save to {}'.format(result_path))
                        
            if args.img_show:
                cv2.imshow("full post process result", img_p)
                cv2.waitKeyEx(0)

        # record maps
        if args.coco_map_test is True:
            if boxes is not None:
                for i in range(boxes.shape[0]):
                    co_helper.add_single_record(image_id = int(img_name.split('.')[0]),
                                                category_id = coco_id_list[int(classes[i])],
                                                bbox = boxes[i],
                                                score = round(scores[i], 5).item()
                                                )

    # calculate maps
    if args.coco_map_test is True:
        pred_json = args.model_path.split('.')[-2]+ '_{}'.format(platform) +'.json'
        pred_json = pred_json.split('/')[-1]
        pred_json = os.path.join('./', pred_json)
        co_helper.export_to_json(pred_json)

        from py_utils.coco_utils import coco_eval_with_json
        coco_eval_with_json(args.anno_json, pred_json)

    # release
    model.release()
