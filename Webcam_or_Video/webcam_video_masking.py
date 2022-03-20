import cv2
import numpy as np

def webcam_masking():

    modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "deploy.prototxt.txt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

    ## Initialize Webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ## Capture frame-by-frame
        ret, frame = cap.read()

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),1.0, (300, 300), (104.0, 117.0, 123.0)
            )

        net.setInput(blob)
        faces = net.forward()
        height, width = frame.shape[:2]

        # Set kernel Width and Height of OpenCV Gaussian Blur module. 
        # The larger the resolution, the larger should the kernel dimensions be
        kernel_width = (width//5) if (width//5)%2==1 else (width//5)+1
        kernel_height = (height//5) if (height//5)%2==1 else (height//5)+1

            ## Detect faces in frame
        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.5:
                box = faces[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x, y, x1, y1) = box.astype("int")

                face = frame[y:y1,x:x1]

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

                frame[y:y1,x:x1] = blurred_face

        ## Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    webcam_masking()

