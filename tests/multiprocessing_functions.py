import os
import cv2
import logging
# current_dir = os.getcwd()
# new_dir = current_dir.replace("\\tests", "")
# os.chdir(new_dir)
# from services.images.image_editor import ImageEditor
# from services.faces.face_detector import FaceDetector
# from services.faces.comparators.face_comparator import FaceComparator

def greet(name, greeting) -> str:
    msg = f"[Process: {os.getpid()}] - {greeting} {name}"
    return msg


def init_worker(root):
    global face_detector
    global face_comparator
    global dataset_root

    dataset_root = root
    # logging.info(f"[Process: {os.getpid()}] - Initializing models")
    # face_detector = FaceDetector()
    # face_comparator= FaceComparator()
    # logging.info(f"[Process: {os.getpid()}] - Models initialized")

def compare_faces(instance, dataset_root):
    logging.info(f"[Process: {os.getpid()}] - Comparing {instance[0]} and {instance[1]}")
    img1 = cv2.cvtColor(cv2.imread(os.path.join(dataset_root, instance[0])), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(os.path.join(dataset_root, instance[1])), cv2.COLOR_BGR2RGB)
    pred1 = face_detector.detect(img1)
    pred2 = face_detector.detect(img2)

    if len(pred1) != 0 and len(pred2) != 0:
        img1 = ImageEditor.crop(img1, pred1[0].bounding_box)
        img2 = ImageEditor.crop(img2, pred2[0].bounding_box)
        return face_comparator.compare(img1, img2).distance
    else:
        return -1