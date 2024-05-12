import argparse
import cv2
import numpy as np
import onnxruntime as rt
import time
from PIL import Image
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--img_path', required=True)
    return parser.parse_args()

def main():
    # parse arguments
    args = parse_args()

    # create the session
    sess = rt.InferenceSession(args.model_path)

    # print the input and output node counts
    num_input_nodes = len(sess.get_inputs())
    num_output_nodes = len(sess.get_outputs())
    print(f"Number of input nodes: {num_input_nodes}")
    print(f"Number of output nodes: {num_output_nodes}")

    # get the image tensor shape
    image_shape = sess.get_inputs()[0].shape
    print(f"Image tensor shape: {image_shape}")

    # get the probability tensor shape and name
    prob_shape = sess.get_outputs()[0].shape
    print(f"Probability tensor shape: {prob_shape}")

    # get the bounding box tensor shape
    bbox_shape = sess.get_outputs()[1].shape
    print(f"Bounding box tensor shape: {bbox_shape}")

    # load the input image
    img = cv2.imread(args.img_path)
    if img is None:
        print(f"Could not read the image: {args.img_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # preprocess the input image
    img_resized = cv2.resize(img, (640, 640),interpolation=cv2.INTER_LINEAR)

    #transopose the image
    img_resized_tensor = img_resized.transpose((2, 0, 1))
    img_resized_tensor = np.ascontiguousarray(img_resized_tensor, dtype=np.float32)
    print(f"Image resized shape: {img_resized_tensor.shape}")

    #i=150
    #print the first 3 values of the image memory
    #print(f"Values: {img_resized_tensor[0,0,0], img_resized_tensor[1,0,0], img_resized_tensor[2,0,0]}")
    #print(f"Values: {img_resized_tensor[0,0,i], img_resized_tensor[1,0,i], img_resized_tensor[2,0,i]}")
    
    # convert the input image to a tensor
    #input_tensor = img_resized.reshape(image_shape)
    #print(f"Input tensor shape: {input_tensor.shape}")
    input_tensor = torch.from_numpy(img_resized_tensor)
    input_tensor = input_tensor[None]
    input_tensor = input_tensor.to('cpu')
    image_np = np.asarray(input_tensor.cpu())
    # run the inference
    start = time.time()
    output_tensors = sess.run(None, {sess.get_inputs()[0].name: image_np})
    end = time.time()

    # print the inference time
    print(f"Inference time: {end - start} ms")

    # extract the probability tensor
    prob_tensor_values = output_tensors[0]

    # extract the bounding box tensor
    bbox_tensor_values = output_tensors[1]

    print(f"Probability tensor values shape: {prob_tensor_values.size, }")
    print(f"final: {(prob_shape[1]-1) * prob_shape[2] + prob_shape[2]}")
    
    for i in range(prob_shape[1]):
        for j in range(prob_shape[2]):
            if prob_tensor_values[0, i, j] > 0.6:
                print(f"Probability tensor values: {prob_tensor_values[0, i, j]}")

    # perform non-maximum suppression
    bboxes = nms_iou(bbox_tensor_values, prob_tensor_values, bbox_shape[1], prob_shape[2], 0.5, 0.2)
    # draw the bounding boxes
    for bbox in bboxes:
        cv2.rectangle(img_resized, (int(bbox.x), int(bbox.y)), (int(bbox.x + bbox.w), int(bbox.y + bbox.h)), (0, 255, 0), 2)
        #cv2.putText(img, str(bbox.class_id), (bbox.x, bbox.y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # save the output image
    
    ok = cv2.imwrite("output2.jpg", img_resized)
    print(f"Output image saved as output.jpg ({ok})")

from typing import List
from collections import namedtuple

bbox_t = namedtuple('bbox_t', ['x', 'y', 'w', 'h', 'score', 'class_id'])

def nms_iou(boxes: List[float], scores: List[float], num_boxes: int, num_classes: int, iou_threshold: float, score_threshold: float) -> List[bbox_t]:

    # create a list of bbox_t objects
    bboxes = []
    for i in range(num_boxes):
        bbox = bbox_t(boxes[0, i, 0], boxes[0, i, 1], boxes[0, i, 2], boxes[0, i, 3], 0.0, 0)
        # find the class with the highest score
        max_score = 0.0
        class_id = 0
        for j in range(num_classes):
            if scores[0, i, j] > max_score:
                max_score = scores[0, i, j]
                class_id = j
        bbox = bbox._replace(score=max_score, class_id=class_id)
        bboxes.append(bbox)

    # sort the bboxes by score
    bboxes.sort(key=lambda x: x.score, reverse=True)

    # apply non-maximum suppression
    nms_bboxes = []
    for i in range(len(bboxes)):
        keep = True
        # check negative coordinates
        if bboxes[i].w < 0 or bboxes[i].h < 0 or bboxes[i].x < 0 or bboxes[i].y < 0:
            continue
        # check score threshold
        if bboxes[i].score < score_threshold:
            continue
        for j in range(len(nms_bboxes)):
            if keep:
                overlap = min(bboxes[i].x + bboxes[i].w, nms_bboxes[j].x + nms_bboxes[j].w) - max(bboxes[i].x, nms_bboxes[j].x)
                overlap *= min(bboxes[i].y + bboxes[i].h, nms_bboxes[j].y + nms_bboxes[j].h) - max(bboxes[i].y, nms_bboxes[j].y)
                iou = overlap / (bboxes[i].w * bboxes[i].h + nms_bboxes[j].w * nms_bboxes[j].h - overlap)
                if iou > iou_threshold:
                    keep = False
        if keep:
            nms_bboxes.append(bboxes[i])

    return nms_bboxes

if __name__ == "__main__":
    main()