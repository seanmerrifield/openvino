import argparse
import cv2
from sys import platform
from pathlib import Path

from inference import Inference
from helpers import preprocessing


# Get correct CPU extension
if platform == "linux" or platform == "linux2":
    CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
    CODEC = 0x00000021
elif platform == "darwin":
    CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib"
    CODEC = cv2.VideoWriter_fourcc('M','J','P','G')
else:
    print("Unsupported OS.")
    exit(1)

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    t_desc = "The model type"
    m_desc = "The location of the model XML file"
    i_desc = "The location of the input file"
    o_desc = "The location of the output file"
    d_desc = "The device name, if not 'CPU'"

    ### TODO: Add additional arguments and descriptions for:
    ###       1) Different confidence thresholds used to draw bounding boxes
    ###       2) The user choosing the color of the bounding boxes
    l_desc = "The confidence threshold for drawing bounding boxes"
    c_desc = "The color of the bounding boxes"


    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-t", help=t_desc, required=True)
    required.add_argument("-m", help=m_desc, required=True)
    optional.add_argument("-i", help=i_desc, default=None)
    optional.add_argument("-o", help=i_desc, default=None)
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-l", help=l_desc, default=0.5)
    optional.add_argument("-c", help=c_desc, default="green")
    args = parser.parse_args()

    return args


def infer_on_video(args):
    ### TODO: Initialize the Inference Engine based on the model type
    inference = Inference()
    net = inference.get_network(args.t)

    ### TODO: Load the network model into the IE
    ### CPU extension is not needed if Openvino 2020+ is used
    net.load_model(args.m, args.d, cpu_extension=None)

    # Get and open video capture
    if args.i is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return
    else:
        cap = cv2.VideoCapture(args.i)
        cap.open(args.i)


    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Create a video writer for the output video
    # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
    # on Mac, and `0x00000021` on Linux
    if args.o is not None:
        out = cv2.VideoWriter(args.o, CODEC, 12, (width,height))
        
    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break

        ### TODO: Pre-process the frame
        input_shape = net.get_input_shape()
        
        image = preprocessing(frame, height=input_shape[2], width=input_shape[3])

        ### TODO: Perform inference on the frame
        net.async_inference(image)
        while True:
            status = net.wait()
            if status == 0: break


        ### TODO: Get the output of inference
        output = net.extract_output()

        ### TODO: Update the frame to include detected bounding boxes
        frame = net.postprocess_output(frame, output)


        # Write out the frame
        if args.o is not None:
            out.write(frame)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        key_pressed = cv2.waitKey(60)

        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the out writer, capture, and destroy any OpenCV windows
    if args.o is not None:
        out.release()

    cap.release()
    cv2.destroyAllWindows()


def main():
    args = get_args()
    infer_on_video(args)


if __name__ == "__main__":
    main()
