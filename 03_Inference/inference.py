from network import VehicleDetection, HumanPose


'''
The Inference class is a broker that returns the
appropriate Network subclass, based on the type of 
network that is requested
'''

class Inference:

    def __init__(self):
        pass


    def get_network(self, typ:str):
        if typ.lower() == "vehicle_detection":
            return VehicleDetection()
        elif typ.lower() == "human_pose":
            return HumanPose()
        else:
            raise Exception(f"Network type {typ} is not supported")

