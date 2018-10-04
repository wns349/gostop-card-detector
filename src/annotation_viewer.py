import os
import cv2
import xml.etree.ElementTree as ET


def _main(args):
    xmls = [f for f in os.listdir(args["xml_dir"]) if f.lower().endswith(".xml")]
    for xml in xmls:
        root = ET.parse(os.path.join(args["xml_dir"], xml)).getroot()
        filename = root.find("filename").text
        img = cv2.imread(os.path.join(args["img_dir"], filename))
        if img is None:
            continue

        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            xmax = int(bndbox.find("xmax").text)
            ymin = int(bndbox.find("ymin").text)
            ymax = int(bndbox.find("ymax").text)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            cv2.putText(img, obj.find("name").text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 0), 2)
        cv2.imshow("image", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", dest="img_dir", help="Path to image directory")
    parser.add_argument("--xml_dir", dest="xml_dir", help="Path to annotation directory")
    _main(vars(parser.parse_args()))
