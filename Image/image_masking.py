import argparse
import logging
import cv2
import imutils
import face_detection
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def image_masking(image_path:str)->np.array:

    detector = face_detection.build_detector(
        "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3
        )

    frame = cv2.imread(image_path)

    d1, d2, d3 = frame.shape

    # Set kernel Width and Height of OpenCV Gaussian Blur module. 
    # The larger the resolution, the larger should the kernel dimensions be
    kernel_width = (d2//5) if (d2//5)%2==1 else (d2//5)+1
    kernel_height = (d1//5) if (d1//5)%2==1 else (d1//5)+1

    w = 300
    h = d1/d2*w

    frame2 = imutils.resize(frame,width=w)
    detections = detector.detect(frame2)[:, :5]

    for face in detections:
        
        x1,y1,x2,y2,conf = [int(i) for i in face]

        Y1 = int(y1/h*d1)
        Y2 = int(y2/h*d1)
        X1 = int(x1/w*d2)
        X2 = int(x2/w*d2)
        
        face = frame[Y1:Y2,X1:X2]

        # create a mask image of the same shape face, filled with 0s (black color)
        mask = np.zeros_like(face)
        rows, cols,_ = mask.shape

        # create a white filled ellipse
        mask=cv2.ellipse(
            mask, center=(int(cols/2), int(rows/2)), axes=(int(cols/2),int(rows/2)), 
            angle=0, startAngle=0, endAngle=360, color=(255,255,255), thickness=-1
            )

        blur = cv2.GaussianBlur(face, (kernel_width, kernel_height), 0)
        blurred_face = np.where(mask==np.array([255, 255, 255]), blur, face)

        frame[Y1:Y2,X1:X2] = blurred_face

    return frame


if __name__=='__main__':

    parser = argparse.ArgumentParser(description = "Identity Masking of Image")
    parser.add_argument('--image_path')
    parser.add_argument('--output_path',default='masked_image.png')
    args, unknown = parser.parse_known_args()

    logger.info("Starting identity masking of image...")
    frame = image_masking(args.image_path)

    logger.info("Writing masked image to file...")
    cv2.imwrite(args.output_path,frame)

    logger.info("Completed!")