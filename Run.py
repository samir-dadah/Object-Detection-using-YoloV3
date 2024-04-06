 
photo_filename ='/content/drive/MyDrive/image.jpg'
    
    # load picture with old dimensions
image, image_w, image_h = load_image_pixels(photo_filename, (WIDTH, HEIGHT))
print(image_w,image_h) 
    # Predict image
yhat = model1.predict(image)
    
    # Create boxes
boxes = list()
for i in range(len(yhat)):
        # decode the output of the network
    boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, HEIGHT, WIDTH)
 
    # correct the sizes of the bounding boxes for the shape of the image
correct_yolo_boxes(boxes, image_h, image_w, HEIGHT, WIDTH)
 
    # suppress non-maximal boxes
do_nms(boxes, 0.5)
 
    # define the labels (Filtered only the ones relevant for this task, which were used in pretraining the YOLOv3 model)
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck","boat"]
 
    # get the details of the detected objects
v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
 
    # summarize what we found
for i in range(len(v_boxes)):
 
    print(v_labels[i], v_scores[i])
 
    # draw what we found
draw_boxes(photo_filename, v_boxes, v_labels, v_scores)
