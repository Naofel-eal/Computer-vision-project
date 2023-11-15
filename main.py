import cv2
from classes.model import Model

def main():
    model = Model()
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        #cv2.imshow("preview", frame)
        model.detect(frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

if __name__ == "__main__":
    main()