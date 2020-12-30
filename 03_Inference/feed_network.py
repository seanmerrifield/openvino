import argparse
from openvino.inference_engine import IENetwork, IECore

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
DEVICE = "CPU"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Load an IR into the Inference Engine")
    # -- Create the descriptions for the commands
    c_desc = "CPU extension file location, if applicable"
    m_desc = "The location of the model XML file"

    # -- Create the arguments
    parser.add_argument("-c", help=c_desc, default=None)
    parser.add_argument("-m", help=m_desc)
    args = parser.parse_args()

    return args


def load_to_IE(model_xml, cpu_ext):
    ### TODO: Load the Inference Engine API
    core = IECore()


    ### TODO: Load IR files into their related class
    model_bin = model_xml.split(".")[0] + ".bin"
    net = core.read_network(model=model_xml, weights=model_bin)

    ### TODO: Add a CPU extension, if applicable. It's suggested to check
    ###       your code for unsupported layers for practice before 
    ###       implementing this. Not all of the models may need it.
    if cpu_ext is not None:
        core.add_extension(cpu_ext, DEVICE)

    ### TODO: Get the supported layers of the network
    layers_map = core.query_network(network=net, device_name=DEVICE)



    ### TODO: Check for any unsupported layers, and let the user
    ###       know if anything is missing. Exit the program, if so.
    
    for layer, device in layers_map.items():
        if DEVICE not in device:
            return
    ### TODO: Load the network into the Inference Engine
    core.load_network(net, DEVICE)

    print("IR successfully loaded into Inference Engine.")

    return


def main():
    args = get_args()
    load_to_IE(args.m, args.c)


if __name__ == "__main__":
    main()
