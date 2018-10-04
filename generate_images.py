import os
import random
import cv2
import numpy as np
from imgaug import augmenters as iaa
import string
import xml.etree.ElementTree as ET

iaa_object = iaa.Sequential(
    [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Sometimes(0.5, iaa.Affine(rotate=(-45, 45), cval=(0, 255), mode="constant")),
        iaa.Sometimes(0.5, iaa.OneOf([iaa.GaussianBlur(0, 3.0), iaa.AverageBlur(k=(2, 7)), iaa.MedianBlur(k=(3, 11))])),
        iaa.Add((-10, 10), per_channel=0.5),
        iaa.AddToHueAndSaturation((-20, 20)),
    ],
    random_order=True
)


def _get_random_background(backgrounds, target_size=None):
    img_bg = None
    while img_bg is None:
        b = random.sample(backgrounds, 1)
        img_bg = cv2.imread(b[0])

    if target_size is not None:  # randomly crop image to target size
        h, w = img_bg.shape[0:2]
        if h < target_size[0]:
            img_bg = np.pad(img_bg, [(0, target_size[0] - h), (0, 0), (0, 0)], mode="constant")
            h = target_size[0]
        if w < target_size[1]:
            img_bg = np.pad(img_bg, [(0, 0), (0, target_size[1] - w), (0, 0)], mode="constant")
            w = target_size[1]
        oy = int(random.uniform(0, h - target_size[0]))
        ox = int(random.uniform(0, w - target_size[1]))
        img_bg = img_bg[oy:oy + target_size[0], ox:ox + target_size[1], :]
    return img_bg


def _get_random_objects(objects, min=1, max=5, eps=0.5):
    r = []
    for obj, label in random.sample(objects, random.randint(min, max)):
        img_obj = cv2.imread(obj)
        # augment image
        if random.uniform(0., 1.) < eps:
            img_obj = iaa_object.augment_image(img_obj)
        r.append((img_obj, label))
    return r


def _add_objects_to_background(img_bg, objects, bg_obj_ratio=0.2):
    img_bg_h, img_bg_w = img_bg.shape[0:2]
    bounding_boxes = []
    for obj, label in objects:
        obj_h, obj_w = obj.shape[0:2]
        # resize image, if necessary
        side_ratio, side_obj, side_img = obj_h / img_bg_h, obj_h, img_bg_h
        if obj_w / img_bg_w > side_ratio:
            side_ratio, side_obj, side_img = obj_w / img_bg_w, obj_w, img_bg_w
        if side_ratio > bg_obj_ratio:
            obj = cv2.resize(obj, None, fx=bg_obj_ratio * side_img / side_obj, fy=bg_obj_ratio * side_img / side_obj)
        obj_h, obj_w = obj.shape[0:2]
        oy = int(random.uniform(0, img_bg_h - obj_h))
        ox = int(random.uniform(0, img_bg_w - obj_w))
        img_bg[oy:oy + obj_h, ox:ox + obj_w, :] = obj
        bounding_boxes.append(BoundingBox(ox, oy, ox + obj_w, oy + obj_h, label))
    return img_bg, bounding_boxes


def _non_maximum_suppression(bounding_boxes, threshold=0.3, min_area=100):
    if bounding_boxes is None or len(bounding_boxes) == 0:
        return []

    bounding_boxes.reverse()
    for i in range(0, len(bounding_boxes)):
        upper_bb = bounding_boxes[i]
        for j in range(i + 1, len(bounding_boxes)):
            lower_bb = bounding_boxes[j]
            xi_min = max(upper_bb.x_min, lower_bb.x_min)
            yi_min = max(upper_bb.y_min, lower_bb.y_min)
            xi_max = min(upper_bb.x_max, lower_bb.x_max)
            yi_max = min(upper_bb.y_max, lower_bb.y_max)
            area_inter = max(0, (yi_max - yi_min)) * max(0, (xi_max - xi_min))
            iou = area_inter / (upper_bb.area() + lower_bb.area() - area_inter + 1e-9)
            # instead of removing the object, accumulate iou score
            lower_bb.iou += iou
    return [b for b in bounding_boxes if b.iou < threshold]


def _make_annotation(image, image_filename, bounding_boxes):
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = os.path.dirname(os.path.abspath(image_filename))
    ET.SubElement(root, "filename").text = os.path.basename(image_filename)
    ET.SubElement(root, "path").text = os.path.abspath(image_filename)
    ET.SubElement(ET.SubElement(root, "source"), "database").text = "Unknown"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "height").text = str(image.shape[0])
    ET.SubElement(size, "width").text = str(image.shape[1])
    ET.SubElement(size, "depth").text = str(image.shape[2])
    # ET.SubElement(root, "segmented").text = "0"
    for bb in bounding_boxes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = str(bb.label)
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(bb.x_min)
        ET.SubElement(bndbox, "xmax").text = str(bb.x_max)
        ET.SubElement(bndbox, "ymin").text = str(bb.y_min)
        ET.SubElement(bndbox, "ymax").text = str(bb.y_max)
    return root


def _main(args):
    objects = [(os.path.join(args["obj_dir"], f), os.path.splitext(f)[0].split("_")[0])
               for f in os.listdir(args["obj_dir"]) if f.lower().endswith(".jpg")]  # (img_path, label)
    backgrounds = [os.path.join(args["bg_dir"], f) for f in os.listdir(args["bg_dir"]) if f.lower().endswith(".jpg")]
    print("Found {} objects / {} backgrounds".format(len(objects), len(backgrounds)))

    if args["prefix"] is None:
        prefix = ''.join(random.choice(string.ascii_uppercase) for _ in range(3))
    else:
        prefix = args["prefix"]

    for idx in range(args["limit"]):
        objs = _get_random_objects(objects, min=2, max=8)
        img_bg = _get_random_background(backgrounds, target_size=(416, 416))
        img, bounding_boxes = _add_objects_to_background(img_bg.copy(), objs, bg_obj_ratio=0.5)
        bounding_boxes = _non_maximum_suppression(bounding_boxes, threshold=0.4)

        # save image
        img_name = os.path.join(args["out_dir"], "{}_{}.jpg".format(prefix, idx))
        xml_name = os.path.join(args["out_dir"], "{}_{}.xml".format(prefix, idx))
        root = _make_annotation(img, img_name, bounding_boxes)
        cv2.imwrite(img_name, img)
        ET.ElementTree(root).write(xml_name)
        print("{}/{} created.".format(img_name, xml_name))


class BoundingBox(object):
    def __init__(self, x_min, y_min, x_max, y_max, label):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.label = label
        self.iou = 0.

    def area(self):
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def __str__(self):
        return "TL: ({},{}) / BR: ({},{})".format(self.x_min, self.y_min, self.x_max, self.y_max)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_dir", dest="obj_dir", help="Path to object image directory")
    parser.add_argument("--bg_dir", dest="bg_dir")
    parser.add_argument("--out_dir", dest="out_dir", help="Path to output directory", default="./")
    parser.add_argument("--limit", dest="limit", help="Total number of images to generate", default=5, type=int)
    parser.add_argument("--prefix", dest="prefix", help="Image prefix", default=None)
    args = parser.parse_args()
    _main(vars(args))
