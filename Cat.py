import cv2 # pip install opencv-python
import numpy as np # pip install numpy

#Get the video
video = cv2.VideoCapture('VideoSource/Cat.mp4')
#Read width and height of video frame
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
threshold = 20.

#Create video result
writer = cv2.VideoWriter('Result/Cat.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 25, (width, height))
ret, frame1 = video.read()
prev_frame = frame1

a = 0 #Total frames are trained
b = 0 #Unique frames are kept
c = 0 #Common frames

#read the model
protopath = 'MobileNetSSD_deploy.prototxt'
modelpath = 'MobileNetSSD_deploy.caffemodel'

#Use model DNN, framework Caffe
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

#21 topics
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

while ret == True:
    #Read the frame of video
    ret, frame = video.read()
    #check if have frame? -> if not out the loop
    if ret == False: break
    #read height-2 and width-2 of frame
    (H, W) = frame.shape[:2]

    #facilitate image preprocessing for deep learning classification
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

    detector.setInput(blob)
    object_detections = detector.forward()

    for i in np.arange(0, object_detections.shape[2]):
        confidence = object_detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(object_detections[0, 0, i, 1])

            #if not a cat delete that frame
            #You can change the tag here if you have train model
            if CLASSES[idx] != "cat":
                continue
            
            #detect the subject
            person_box = object_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = person_box.astype("int")

            #Drawing bounding box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

            #Check if it a Unique frame? Save it
            if (((np.sum(np.absolute(frame - prev_frame)) / np.size(frame)) > threshold)):
                writer.write(frame)
                prev_frame = frame
                a += 1
            else:
                prev_frame = frame
                b += 1

    #show video when training
    cv2.imshow("Application", frame)
    c += 1

    #stop if you want -> press "q"
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

#Print result
print("Total frames: ", c)
print("Unique frames: ", a)
print("Common frames: ", b)
#save result
video.release()
writer.release()
cv2.destroyAllWindows()