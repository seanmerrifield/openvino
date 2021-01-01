
'''
Contains code for working with the Inference Engine.
You'll learn how to implement this code and more in
the related lesson on the topic.
'''

import os
import sys
import time
import logging as log

from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np

from abc import abstractmethod
 

class Network:
    '''
    Load and store information for working with the Inference Engine,
    and any loaded models.
    '''

    def __init__(self):
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None


    def load_model(self, model, device="CPU", cpu_extension=None):
        '''
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        '''
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Initialize the plugin
        self.plugin = IECore()

        # Add a CPU extension, if applicable
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)

        # Read the IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network, device)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        return


    def get_input_shape(self):
        '''
        Gets the input shape of the network
        '''
        return self.network.inputs[self.input_blob].shape


    def async_inference(self, image):
        '''
        Makes an asynchronous inference request, given an input image.
        '''
        ### TODO: Start asynchronous inference
        return self.exec_network.start_async(request_id=0, inputs={self.input_blob: image})


    def wait(self):
        '''
        Checks the status of the inference request.
        '''
        ### TODO: Wait for the async request to be complete
        status = self.exec_network.requests[0].wait(-1)
        return status

    @abstractmethod
    def extract_output(self):
        '''
        Returns a list of the results for the output layer of the network.
        '''
        ### Return the outputs of the network from the output_blob
        return self.exec_network.requests[0].outputs[self.output_blob]


    @abstractmethod
    def postprocess_output(self, image, output):
        '''
        Returns a list of the results for the output layer of the network.
        ''' 
        return image

    

class VehicleDetection(Network):

    BOX_COLOR = "GREEN"
    THRESHOLD = 0.5

    def __init__(self):
        super().__init__()


    def postprocess_output(self, image, output):
        '''
        Using the model type, input image, and processed output,
        creates an output image showing the result of inference.
        '''
        
        color = self.BOX_COLOR.lower()
        threshold = self.THRESHOLD

        color_channels = {"blue": (255,0,0), "green": (0,255,0), "red": (0,0,255)}
        if color not in color_channels: raise Exception(f"Color: {color} is not a valid color option (must be blue, green, or red)")

        #Remove excess dimensions so that it is a list of bounding boxes
        output = output.squeeze()

        for image_id, label, conf, x_min, y_min, x_max, y_max in output:

            # Skip if confidence level is below 0.5
            if conf < threshold: continue

            x_min = int(x_min * image.shape[1])
            x_max = int(x_max * image.shape[1])
            y_min = int(y_min * image.shape[0])
            y_max = int(y_max * image.shape[0])
            image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color_channels[color], 2) 


        return image
        




class HumanPose(Network):

    def __init__(self):
        super().__init__()

    def load_model(self, model, device="CPU", cpu_extension=None):
        super().load_model(model, device, cpu_extension)

        self.output_blob = list(self.network.outputs)[1]


    def extract_output(self):
        '''
        Returns a list of the results for the output layer of the network.
        '''
        ### TODO: Return the outputs of the network from the output_blob
        return self.exec_network.requests[0].outputs[self.output_blob]

    
    def postprocess_output(self, image, output):
        """ Postprocess output results for human pose estimation

        Args:
            image (np.array): Input image with shape (height (h), width (w), channels (c))
            output (np.array): Inference output with shape (batch (b), classes (n), inf_height (hi), inf_width (wi))

        Returns:
            [type]: [description]
        """

        # Remove final part of output not used for heatmaps
        # Shape (b, n, hi, wi)
        output = output.squeeze()

        # Copy initial output prior to postprocessing
        # Shape (n, hi, wi)
        init_output = np.copy(output)

        # Resize output to original input size
        # Shape (n, hi, wi)
        w = image.shape[1]
        h = image.shape[0]

        output = np.zeros([init_output.shape[0], h, w])
        for i, img in enumerate(init_output):
            output[i] = cv2.resize(img, image.shape[0:2][::-1])


        # Remove final part of output not used for heatmaps
        output = output[:-1]

        # For each class, if confidence threshold set value at locations where confidence > 0.5
        for c in range(len(output)):
            output[c] = np.where(output[c]>0.5, 255, 0)

        
        # Sum along the "class" axis
        # Shape (h, w)
        output = np.sum(output, axis=0)
        #output = output[0]

        # Get semantic mask
        # Shape (h, w, c)
        pose_mask = self.get_mask(output)

        # Combine with original image
        # Shape (h, w, c)
        image = cv2.add(image, pose_mask)
        image = np.where(image > 255, 255, image)
        return image


    def get_mask(self, processed_output):
        '''
        Given an input image size and processed output for a semantic mask,
        returns a masks able to be combined with the original image.
        '''
        # Create an empty array for other color channels of mask
        empty = np.zeros(processed_output.shape)
        # Stack to make a Green mask where text detected
        mask = np.dstack((empty, processed_output, empty))

        return mask.astype(np.uint8)
