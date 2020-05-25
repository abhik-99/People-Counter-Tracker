"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


#Taken from https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
#implements non-max suppression
# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh = 0.4):
    import numpy as np
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes	
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,3]
    y1 = boxes[:,4]
    x2 = boxes[:,5]
    y2 = boxes[:,6]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    
    # keep looping while some indexes still remain in the indexes list
    
    while len(idxs) > 0:
        
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
        
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")



def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
#     client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    print("OPENCV", cv2.__version__)
    import numpy as np
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    
    network = infer_network.load_model(model= args.model, device = args.device, cpu_extension = args.cpu_extension)
    ### TODO: Handle the input stream ###
    if sys.platform == "linux" or sys.platform == "linux2":
        CODEC = 0x00000021
    elif sys.platform == "darwin":
        CODEC = cv2.VideoWriter_fourcc('M','J','P','G')
    else:
        print("Unsupported OS.")
        exit(1)
        
    image_flag = False
    if args.input == 'CAM':
        args.i = 0
    elif args.input.split(".")[-1] == 'jpg' or args.input.split(".")[-1] == 'bmp' or args.input.split(".")[-1] == 'png':
        image_flag = True
    
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    
    if not image_flag:
        out = cv2.VideoWriter('output.mp4', CODEC, 30, (700,600))
        
#     print("INPUT SHAPE of the NETWORK -", infer_network.get_input_shape())
#     print("IMAGE FLAG - ", image_flag)
    tracker_mosse = cv2.TrackerKCF_create() #creating a tracker
    
    net_input_shape = infer_network.get_input_shape()
    
    PERSON_COUNT = 0 #for counting the number of persons
    TRACKER_INIT_FLAG = True
    
    while cap.isOpened():
    ### TODO: Loop until stream is over ###
        
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        
        #starting timer for recording FPS
        timer = cv2.getTimerCount()
        
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        ### TODO: Pre-process the image as needed ###
        img = cv2.resize( frame, (net_input_shape[3], net_input_shape[2]))
        img = img.transpose((2,0,1))
        img = img.reshape(1, *img.shape)
        
        ### TODO: Start asynchronous inference for specified request ###
        if TRACKER_INIT_FLAG:
            
            infer_network.exec_net(img)
            ### TODO: Wait for the result ###
            if infer_network.wait() == 0:
                ### TODO: Get the results of the inference request ###
                output = infer_network.get_output()

                filter_output = [] #filter only the "Person" (class = 1)
                for each in output[0][0]:
                    
                    if( each[1] == 1):
                        if( prob_threshold != None):
                            #if probabilty threshold has been given as argument
                            if( each[2] >= prob_threshold ):
                                filter_output.append([each[0], each[1], each[2] * 100, int(each[3] * net_input_shape[3]), int(each[4] * net_input_shape[2]), int(each[5] * net_input_shape[3]), int(each[6] * net_input_shape[2])])  
                        else:
                            filter_output.append([each[0], each[1], each[2] * 100 , int(each[3] * net_input_shape[3]), int(each[4] * net_input_shape[2]), int(each[5] * net_input_shape[3]), int(each[6] * net_input_shape[2])])

                #filter output now contains the original sized image boundaries
                filter_output = np.array(filter_output, dtype = 'int')
                filter_output = non_max_suppression_fast(filter_output, 0.3)

                o_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))

                CURRENT_COUNT = len(filter_output)

                for each in filter_output:
                    if PERSON_COUNT == 0:
                        TRACKER_INIT_FLAG = not (tracker_mosse(o_frame, each[3:]))
                        PERSON_COUNT += 1
                        o_frame = cv2.rectangle( o_frame, (each[3],each[4]), (each[5], each[6]), (0,255,0), 2, 1)
                        
                        cv2.putText( o_frame, "PERSON_COUNT" + str(PERSON_COUNT), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                        cv2.putText(o_frame, "FPS : " + str(int(fps)), (10,300), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
            else:
                #Tracker has been initialized, now we just need to track the object
                o_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
                ret, box = tracker_mosse.update(o_frame)
                if not ret:
                    TRACKER_INIT_FLAG = True
                    cv2.putText(frame, "Tracking failure detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                else:
                    start = (int(bbox[0]), int(bbox[1]))
                    end = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    o_frame = cv2.rectangle(o_frame, start, end, (0,255,0), 2, 1)
                    cv2.putText( o_frame, "PERSON_COUNT" + str(PERSON_COUNT), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                    cv2.putText(o_frame, "FPS : " + str(int(fps)), (10,300), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)


            output = cv2.resize(o_frame, (768, 432))

            ### TODO: Extract any desired stats from the results ###
            
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            
#             client.publish("person", json.dumps({ "count": CURRENT_COUNT, "total": PERSON_COUNT}))
#             if PERSON_TRACKER.shape[0] > 1:
#                 mean = np.mean(PERSON_TRACKER[:, 1])
#             elif PERSON_TRACKER.shape == (1,5) :
# #                 print(PERSON_TRACKER)
#                 mean = PERSON_TRACKER[0,1] 
#             else:
#                 mean = 0
                
#             ### Topic "person/duration": key of "duration" ###
#             client.publish("person/duration", json.dumps({"duration": mean}))

        ### TODO: Send the frame to the FFMPEG server ###
#         sys.stdout.buffer.write(output)
#         sys.stdout.flush()
        
        ### TODO: Write an output image if `single_image_mode` ###
        if not image_flag:
            out.write(output)
        else:
            cv2.imwrite('output_image.jpg', output)
        
        if key_pressed == 27:
            break
    
    if not image_flag:
        out.release()
    cap.release()
    cv2.destroyAllWindows()
    
#     client.disconnect()
    

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
