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

MODEL_NAME = "ssd_mobilenet_v1_coco"

FRAME_WIDTH = 768
FRAME_HEIGHT = 432

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
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
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
        out = cv2.VideoWriter('output.mp4', CODEC, 30, (FRAME_WIDTH, FRAME_HEIGHT))
    
    net_input_shape = infer_network.get_input_shape()
    
    person_count = 0 #for counting the number of persons
    PERSON_DURATION = 0 #for the duration of time the person spends in the frames
    person_tracker = np.array([]) #for trackin the person 0->person_count, 1-> PERSON_DURATION, 2-> PERSON_TRACKING , 3-> MISSING FPS, 4-> had started tracking from 
    person_centroids = np.array([]) # keeps all the centroids of the people tracked.
    frames_elapsed = 0
    no_longer_tracking = [] #Keeping a list of all those people no longer being tracked
    mean = 0
    p_mean = 0
    
    
    while cap.isOpened():
    ### TODO: Loop until stream is over ###
        
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        
        #Getting Frames elapsed and FPS
        frames_elapsed += 1
        fps = cap.get(cv2.CAP_PROP_FPS) #Please note that this might not work with Webcam

        ### TODO: Pre-process the image as needed ###
        img = cv2.resize( frame, (net_input_shape[3], net_input_shape[2]))
        img = img.transpose((2,0,1))
        img = img.reshape(1, *img.shape)
        
        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(img)
        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            ### TODO: Get the results of the inference request ###
            output = infer_network.get_output()
            
            filter_output = [] #filter only the "Person" (class = 1)
            for each in output[0][0]:
#                 print("NETWORK OUTPUT", each)
                if( each[1] == 1):
                    if( prob_threshold != None):
                        #if probabilty threshold has been given as argument
                        if( each[2] >= prob_threshold ):
#                             filter_output.append([each[0], each[1], each[2] * 100, int(each[3] * net_input_shape[3]), int(each[4] * net_input_shape[2]), int(each[5] * net_input_shape[3]), int(each[6] * net_input_shape[2])])
                            filter_output.append([each[0], each[1], each[2] * 100, int(each[3] * FRAME_WIDTH), int(each[4] * FRAME_HEIGHT), int(each[5] * FRAME_WIDTH), int(each[6] * FRAME_HEIGHT)])
                    else:
                        if( each[2] >= 0.6):
                            #Default prob threshold is 0.6
#                             filter_output.append([each[0], each[1], each[2] * 100 , int(each[3] * net_input_shape[3]), int(each[4] * net_input_shape[2]), int(each[5] * net_input_shape[3]), int(each[6] * net_input_shape[2])])
                            filter_output.append([each[0], each[1], each[2] * 100 , int(each[3] * FRAME_WIDTH), int(each[4] * FRAME_HEIGHT), int(each[5] * FRAME_WIDTH), int(each[6] * FRAME_HEIGHT)])
                        
            #filter output now contains the original sized image boundaries
            filter_output = np.array(filter_output, dtype = 'int')
            filter_output = non_max_suppression_fast(filter_output, 0.3)
            o_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            tracked = [] #To get the IDs of the people tracker in this frame.
            CURRENT_COUNT = len(filter_output)


            #LOGIC:-            
            # 1. For every detection, calculate the centroid of the detection
            # 2. If the centroid is close to one of the tracking centroids, then the person must be still in the frame.
            # 3. Update the matched centroid.
            # 4. If no matching centroid is found for a detection, then it is a new person. The person is added to the person_tracking array.
            # 5. For every person who is being tracked but is not detected in the frame, the specific criteria in line 309
            # 6. A person who is not being tracked has his/her stats removed from person_tracking and person_centroids and put into no_longer_tracking



            for each in filter_output:
                PERSON_CENTROID = ((each[3] + each[5])/2, (each[4] + each[6])/2 )
                person_found_flag = False
                
                if len(person_centroids) > 0 :                    
                    x, y = person_centroids[np.where(person_tracker[:,2] == 1), 0].flatten(), person_centroids[ np.where(person_tracker[:,2] == 1), 1].flatten()
                    distances = ( (x - PERSON_CENTROID[0])**2 + (y - PERSON_CENTROID[1])**2 )**0.5
                    
                    if len(distances) > 0:
                        
                        idx = np.argsort(distances)[0]
                        
                        if distances[idx] <= 85 :
                            #MIN_DISTANCE Criteria satisfied!
                            #if we are still tracking the person
                            o_frame = cv2.rectangle(o_frame, (each[3], each[4]), (each[5], each[6]), (255, 0, 0), 2)
                            o_frame = cv2.circle(o_frame, (int(PERSON_CENTROID[0]), int(PERSON_CENTROID[1])), 2, (0,0,255), 2) #Putting the centroid
                            person_tracker[idx][1] = time.time() - person_tracker[idx][4] #Updating the Duration of Tracking

                            text_1 = "PERSON_ID-"+ str(int(person_tracker[idx][0])+1)
                            text_2 = "LOC-" + str(person_centroids[idx])
                            text_3 = "DUR-" + str(person_tracker[idx][1])

                            cv2.putText(o_frame, text_1 , (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                            cv2.putText(o_frame, text_2 , (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                            cv2.putText(o_frame, text_3 , (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                            person_centroids[idx] = PERSON_CENTROID #updating the person's centroid
                            person_found_flag = True #no new person was encountered
                            tracked.append(person_tracker[idx][0]) #adding idx to the tracked list
                
                if person_found_flag == False :
                    #person was not found so 
                    
                    person = [person_count, 0, 1, 0, time.time()]
                    
                    if len(person_centroids) == 0:
                        person_tracker = np.append(person_tracker, person)
                        person_centroids = np.append(person_centroids, PERSON_CENTROID)
                        person_tracker = np.expand_dims(person_tracker, axis=0)
                        person_centroids = np.expand_dims(person_centroids, axis=0)
                    else:
                        person_tracker = np.vstack([person_tracker, person])
                        person_centroids = np.vstack([person_centroids, PERSON_CENTROID])
                    
                    o_frame = cv2.rectangle(o_frame, (each[3], each[4]), (each[5], each[6]), (255, 0, 0), 2)
                    o_frame = cv2.circle(o_frame, (int(PERSON_CENTROID[0]), int(PERSON_CENTROID[1])), 2, (0,0,255), 2) #putting the centroid
                    text_1 = "PERSON_ID-"+ str(person_count+1) 
                    text_2 = "LOC-" + str(PERSON_CENTROID)
                    text_3 = "DUR-" + str(person[1])
                    
                    cv2.putText(o_frame, text_1 , (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(o_frame, text_2 , (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(o_frame, text_3 , (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    
                    tracked.append(person_count)
                    person_count += 1
                    person_found_flag = True
            
            i=0
            while(i<len(person_tracker)):
                
                if person_tracker[i][0] not in tracked:
                    #if the person was not tracked in this frame
                    if person_tracker[i][2]:
                        #if the person is still being tracked then Missing FPS is incremented
                        person_tracker[i][3] += 1
                        
                    if (person_tracker[i][3] >= 150 and ( abs(person_centroids[i][0] - FRAME_WIDTH) <= 150 or abs(person_centroids[i][1] - FRAME_HEIGHT) <= 50)) or ( abs(person_centroids[i][0] - FRAME_WIDTH) <= 150 or abs(person_centroids[i][1] - FRAME_HEIGHT) <= 50) or person_tracker[i][3] >= 250:
                        #if the person has been missing for more than 100 frames then the person has left so removing him/her form list
                        if person_tracker[i][2]:
                            person_tracker[i][2] = 0
                            person_stats = person_tracker[i]
                            
                            person_centroids = list(person_centroids)
                            person_tracker = list(person_tracker)
                            
                            person_centroid = person_centroids.pop(i)
                            person_stats = person_tracker.pop(i)
                            
                            person_centroids = np.array(person_centroids)
                            person_tracker = np.array(person_tracker)
                            
                            no_longer_tracking.append([person_stats, person_centroid])
                            
                else:
                    person_tracker[i][3] = 0 #Person has been tracked and so his missing frames count is reset
                i += 1
                
                    
            for each_person in range(len(person_tracker)):
                
                if(person_tracker[each_person, 2]):
                    o_frame = cv2.circle(o_frame, (int(person_centroids[each_person][0]), int(person_centroids[each_person][1]) ), 2, (0, 255, 0), 2) #putting the centroid for tracking person
                    cv2.putText(o_frame, "("+str(person_centroids[each_person][0])+", "+str(person_centroids[each_person][1])+")", (int(person_centroids[each_person][0]), int(person_centroids[each_person][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
                    
            for each_person in no_longer_tracking:
                
                o_frame = cv2.circle(o_frame, (int(each_person[1][0]), int(each_person[1][1]) ), 2, (0, 0, 255), 2) #putting the centroid for non-tracking person
                
                cv2.putText(o_frame, "("+str(each_person[1][0])+", "+str(each_person[1][1])+")", (int(each_person[1][0]), int(each_person[1][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
                
            
                
            cv2.putText(o_frame, "Total Frames - " + str(frames_elapsed), (10, FRAME_HEIGHT - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            cv2.putText(o_frame, "FPS - " + str(fps), (10, FRAME_HEIGHT - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            cv2.putText(o_frame, "Inference Time - " + str(frames_elapsed / fps), (10, FRAME_HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            cv2.putText(o_frame, "Total Count - "+str(person_count)+", Current Count - "+ str(len(filter_output)), (10, FRAME_HEIGHT - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)            
            
            cv2.putText(o_frame, "MODEL- "+MODEL_NAME, (10, FRAME_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            
            output = cv2.resize(o_frame, (FRAME_WIDTH, FRAME_HEIGHT))

            ### TODO: Extract any desired stats from the results ###
            
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            client.publish("person", json.dumps({ "count": len(person_centroids), "total": str(person_count)}))
            
            
            
            if len(no_longer_tracking) > 0:
                mean = np.mean([p[1] for p, c in no_longer_tracking])
                
            p_mean = mean
            ### Topic "person/duration": key of "duration" ###
            client.publish("person/duration", json.dumps({"duration": mean}))

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(output)
        sys.stdout.flush()
        
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
    
    client.disconnect()
    

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
