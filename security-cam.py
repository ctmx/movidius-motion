#!/usr/bin/python3

# ****************************************************************************
# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.
# ****************************************************************************

# Detect objects on a LIVE camera feed using
# Intel® Movidius™ Neural Compute Stick (NCS)

import os
import cv2
import sys
import numpy
import ntpath
import argparse
import requests
from videocaptureasync import VideoCaptureAsync
import mvnc.mvncapi as mvnc

from time import localtime, strftime
from datetime import datetime, timedelta
from utils import visualize_output
from utils import deserialize_output

# "Class of interest" - Display detections only if they match this class ID
CLASS_PERSON         = 15

# Detection threshold: Minimum confidance to tag as valid detection
CONFIDANCE_THRESHOLD = 0.60 # 60% confidant

# Variable to store commandline arguments
ARGS                 = None

# OpenCV object for video capture
camera               = None

# Minimum time between notifications
MINTIME              = 10

# Variable to store initial/seed d_last tim
d_last               = datetime.now() - timedelta(seconds=MINTIME)

# ---- Step 1: Open the enumerated device and get a handle to it -------------

def open_ncs_device():

    # Look for enumerated NCS device(s); quit program if none found.
    devices = mvnc.EnumerateDevices()
    if len( devices ) == 0:
        print( "No devices found" )
        quit()

    # Get a handle to the first enumerated device and open it
    device = mvnc.Device( devices[0] )
    device.OpenDevice()

    return device

# ---- Step 2: Load a graph file onto the NCS device -------------------------

def load_graph( device ):

    # Read the graph file into a buffer
    with open( ARGS.graph, mode='rb' ) as f:
        blob = f.read()

    # Load the graph buffer into the NCS
    graph = device.AllocateGraph( blob )

    return graph

# ---- Step 3: Pre-process the images ----------------------------------------

def pre_process_image( frame ):

    # Resize image [Image size is defined by choosen network, during training]
    img = cv2.resize( frame, tuple( ARGS.dim ) )

    # Convert RGB to BGR [OpenCV reads image in BGR, some networks may need RGB]
    if( ARGS.colormode == "rgb" ):
        img = img[:, :, ::-1]

    # Mean subtraction & scaling [A common technique used to center the data]
    img = img.astype( numpy.float16 )
    img = ( img - numpy.float16( ARGS.mean ) ) * ARGS.scale

    return img

# ---- Step 4: Read & print inference results from the NCS -------------------

def infer_image( graph, img, frame ):

    # Load the image as a half-precision floating point array
    graph.LoadTensor( img, 'user object' )

    # Get the results from NCS
    output, userobj = graph.GetResult()

    # Get execution time
    inference_time = graph.GetGraphOption( mvnc.GraphOption.TIME_TAKEN )

    # Deserialize the output into a python dictionary
    output_dict = deserialize_output.ssd(
                      output,
                      CONFIDANCE_THRESHOLD,
                      frame.shape )

    # Print the results (each image/frame may have multiple objects)
    for i in range( 0, output_dict['num_detections'] ):

        # Filter a specific class/category
        if( output_dict.get( 'detection_classes_' + str(i) ) == CLASS_PERSON ):

            # Time
            d = datetime.now()
            cur_time = strftime( "%Y_%m_%d_%H_%M_%S", localtime() )
            print( "Person detected on " + cur_time )

            # Extract top-left & bottom-right coordinates of detected objects
            (y1, x1) = output_dict.get('detection_boxes_' + str(i))[0]
            (y2, x2) = output_dict.get('detection_boxes_' + str(i))[1]

            # Prep string to overlay on the image
            display_str = (
                labels[output_dict.get('detection_classes_' + str(i))]
                + ": "
                + str( output_dict.get('detection_scores_' + str(i) ) )
                + "%" )

            # Overlay bounding boxes, detection class and scores
            frame = visualize_output.draw_bounding_box(
                        y1, x1, y2, x2,
                        frame,
                        thickness=4,
                        color=(255, 255, 0),
                        display_str=display_str )

            # Capture snapshots
            photo = ( os.path.dirname(os.path.realpath(__file__))
                      + "/captures/photo_"
                      + cur_time + ".jpg" )

            cv2.imwrite( photo, frame )

            #IFTT notification
            global d_last
            d_diff = d-d_last

            if (d_diff.seconds > MINTIME) :
                print('\033[31m' + 'Send Notification to IFTTT --- ' + '\033[0m')  # Red text
                #print(d)
                #print(d_last)
                #print(d_diff)
                #print("    ")
                r = requests.post('https://maker.ifttt.com/trigger/rasp_seccam_triggered/with/key/c_6oKb50WdIWAaelvo3EINQ8ZU9ibwxNFJiBV1phPuh', params={"value1":cur_time,"value2":photo,"value3":"none"})
                os.system("rclone copy " + photo + " gdrive:rclone")
                d_last = d

    # If a display is available, show the image on which inference was performed
    if 'DISPLAY' in os.environ:
        cv2.imshow( 'NCS live inference', frame )

# ---- Step 5: Unload the graph and close the device -------------------------

def close_ncs_device( device, graph ):
    graph.DeallocateGraph()
    device.CloseDevice()
    camera.release()
    cv2.destroyAllWindows()

# ---- Main function (entry point for this script ) --------------------------

def main():
    cur_time = strftime( "%Y_%m_%d_%H_%M_%S", localtime() )

    print("**************************************")
    print("NCS app started at " + cur_time)
    print("**************************************")
    filename = "/home/pi/workspace/ncappzoo/apps/security-cam/output/output_" + cur_time + ".log"

    # Open the file with writing permission
    myfile = open(filename, 'w')

    # Write a line to the file
    myfile.write("NCS app started at " + cur_time)

    # Close the file
    myfile.close()

    device = open_ncs_device()
    graph = load_graph( device )

    # Main loop: Capture live stream & send frames to NCS
    while( True ):
        #ret, frame = camera.read()
        #img = pre_process_image( frame )
        #infer_image( graph, img, frame )
        
        camera_1.start()
        ret1, frame1 = camera_1.read()
        camera_2.start()
        ret2, frame2 = camera_2.read()


        if (ret1):
        # Display the resulting frame
            img = pre_process_image( frame1 )
            infer_image( graph, img, frame1 )
            #print("Inferring Cam1 " + cur_time)
            #if 'DISPLAY' in os.environ:
            #    cv2.imshow( 'Camera1', frame1 )


        if (ret2):
        # Display the resulting frame
            img = pre_process_image( frame2 )
            infer_image( graph, img, frame2 )
            #print("Inferring Cam2 " + cur_time)
            #if 'DISPLAY' in os.environ:
            #    cv2.imshow( 'Camera2', frame2 )



        # Display the frame for 5ms, and close the window so that the next
        # frame can be displayed. Close the window if 'q' or 'Q' is pressed.
        if( cv2.waitKey( 5 ) & 0xFF == ord( 'q' ) ):
            break

    camera_1.stop()
    camera_1.stop()
    #camera_1.release()
    #camera_2.release()
    cv2.destroyAllWindows()
    close_ncs_device( device, graph )

# ---- Define 'main' function as the entry point for this script -------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                         description="DIY smart security camera PoC using \
                         Intel® Movidius™ Neural Compute Stick." )

    parser.add_argument( '-g', '--graph', type=str,
                         default='/home/pi/workspace/ncappzoo/caffe/SSD_MobileNet/graph',
                         help="Absolute path to the neural network graph file." )

    parser.add_argument( '-v', '--video', type=int,
                         default=0,
                         help="Index of your computer's V4L2 video device. \
                               ex. 0 for /dev/video0" )

    parser.add_argument( '-l', '--labels', type=str,
                         default='/home/pi/workspace/ncappzoo/caffe/SSD_MobileNet/labels.txt',
                         help="Absolute path to labels file." )

    parser.add_argument( '-M', '--mean', type=float,
                         nargs='+',
                         default=[127.5, 127.5, 127.5],
                         help="',' delimited floating point values for image mean." )

    parser.add_argument( '-S', '--scale', type=float,
                         default=0.00789,
                         help="Absolute path to labels file." )

    parser.add_argument( '-D', '--dim', type=int,
                         nargs='+',
                         default=[300, 300],
                         help="Image dimensions. ex. -D 224 224" )

    parser.add_argument( '-c', '--colormode', type=str,
                         default="bgr",
                         help="RGB vs BGR color sequence. This is network dependent." )

    ARGS = parser.parse_args()

    # Create a VideoCapture object
    # camera = cv2.VideoCapture( ARGS.video )
    camera_1 = VideoCaptureAsync("rtsp://192.168.1.10/1")
    camera_2 = VideoCaptureAsync("rtsp://192.168.1.10/2")


    # Set camera resolution
    # camera.set( cv2.CAP_PROP_FRAME_WIDTH, 352 )
    # camera.set( cv2.CAP_PROP_FRAME_HEIGHT, 288 )
    camera_1.set( cv2.CAP_PROP_FRAME_WIDTH, 352 )
    camera_1.set( cv2.CAP_PROP_FRAME_HEIGHT, 288 )
    camera_2.set( cv2.CAP_PROP_FRAME_WIDTH, 352 )
    camera_2.set( cv2.CAP_PROP_FRAME_HEIGHT, 288 )

    # Load the labels file
    labels =[ line.rstrip('\n') for line in
              open( ARGS.labels ) if line != 'classes\n']

    main()

# ==== End of file ===========================================================
