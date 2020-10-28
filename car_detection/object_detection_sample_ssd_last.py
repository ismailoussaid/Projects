import sys
import os
import glob
from collections import defaultdict
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IECore

path = "C:/Users/Ismail/Documents/Projects/Detect Cars/"
video_input = path + "volta_test_night_0.mp4"

# Opens the Video file
cap= cv2.VideoCapture(video_input)
i=0

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group("Options")
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-m', '--model', help='xml_model_path', required=True, type=str)
    args.add_argument('-i', '--input', help='images_path', required=True, type=str, nargs='+')
    args.add_argument('-c', '--confidence', help='Minimum score to accept a detection', required=False, type=float,
                      default=0.4)
    args.add_argument('-s', '--save', help='Save results to image files', required=False, action='store_true')
    return parser

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    log.info("Loading Inference Engine")
    ie = IECore()
    # --------------------------- 1. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
    model_xml = args.model
    # model_xml = path + "vehicle_detection/vehicle-detection-adas-0002.xml"
    model_bin = model_xml[:-3] + 'bin'
    log.info("Loading network files:\n\t{}\n".format(model_xml))
    net = ie.read_network(model=model_xml, weights=model_bin)
    # -----------------------------------------------------------------------------------------------------

    # ------------- 2. Load Plugin for inference engine and extensions library if specified --------------
    log.info("Device info:")
    device = 'CPU'
    versions = ie.get_versions(device)
    print("{}{}".format(" " * 8, device))
    print("{}MKLDNNPlugin version ......... {}.{}".format(" " * 8, versions[device].major,
                                                          versions[device].minor))
    print("{}Build ........... {}".format(" " * 8, versions[device].build_number))

    supported_layers = ie.query_network(net, "CPU")
    not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    if len(not_supported_layers) != 0:
        log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                  format(device, ', '.join(not_supported_layers)))
        log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                  "or --cpu_extension command line argument")
        sys.exit(1)
    # -----------------------------------------------------------------------------------------------------

    # --------------------------- 3. Read and preprocess input --------------------------------------------

    infos = [*net.input_info]
    print("inputs number: " + str(len(infos)))
    print("input shape: " + str(net.input_info[infos[0]].input_data.shape))
    print("input key: " + infos[0])
    n, c, h, w = net.input_info[infos[0]].input_data.shape

    show = False
    images = []
    images_hw = []
    i=0

    while (cap.isOpened()) and i <10:
        ret, frame = cap.read()
        if ret == False:
            break
        image = frame
        ih, iw = image.shape[:-1]
        off = (ih - h)//2
        i_aspect_ratio = ih / iw
        new_h = int(w * i_aspect_ratio)
        image = cv2.resize(image, (new_h, w))
        mid = ih//2
        crop = image[mid-off:off+mid, :, :]
        images_hw.append(crop.shape[:-1])
        log.info("File added: ")
        log.info("        {} - size {}x{}".format(image, *crop.shape[:-1]))
        if show:
            cv2.imshow('img', image)
            cv2.imshow('crop', crop)
            cv2.waitKey(0)

        crop = crop.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        images.append(crop)
        i += 1

    cap.release()
    cv2.destroyAllWindows()

    # -----------------------------------------------------------------------------------------------------

    # --------------------------- 4. Configure input & output ---------------------------------------------
    # --------------------------- Prepare input blobs -----------------------------------------------------
    log.info("Preparing input blobs")

    out_blob = next(iter(net.outputs))
    input_name = infos[0]
    log.info("Batch size is {}".format(net.batch_size))
    net.input_info[infos[0]].precision = 'U8'

    # --------------------------- Prepare output blobs ----------------------------------------------------
    log.info('Preparing output blobs')

    output_name, output_info = "", net.outputs[next(iter(net.outputs.keys()))]
    for output_key in net.outputs:
        if net.layers[output_key].type == "DetectionOutput":
            output_name, output_info = output_key, net.outputs[output_key]

    if output_name == "":
        log.error("Can't find a DetectionOutput layer in the topology")

    output_dims = output_info.shape
    if len(output_dims) != 4:
        log.error("Incorrect output dimensions for SSD model")
    max_proposal_count, object_size = output_dims[2], output_dims[3]

    if object_size != 7:
        log.error("Output item should have 7 as a last dimension")

    output_info.precision = "FP32"
    # -----------------------------------------------------------------------------------------------------

    # --------------------------- Performing inference ----------------------------------------------------
    log.info("Loading model to the device")
    exec_net = ie.load_network(network=net, device_name=device)
    log.info("Creating infer request and starting inference")

    # -----------------------------------------------------------------------------------------------------
    # --------------------------- Read and postprocess output ---------------------------------------------
    log.info("Processing output blobs")
    output = defaultdict(list)
    for i in range(len(images)):
        data = {input_name: images[i]}
        print(data)
        res = exec_net.infer(inputs=data)
        res = res[out_blob][0][0]
        for number, proposal in enumerate(res):
            if proposal[2] > 0:
                ih, iw = images_hw[i]
                label = np.int(proposal[1])
                confidence = proposal[2]
                xmin = np.int(iw * proposal[3])
                ymin = np.int(ih * proposal[4])
                xmax = np.int(iw * proposal[5])
                ymax = np.int(ih * proposal[6])
                if confidence > args.confidence:
                    print("[{},{}] element, prob = {:.6}    ({},{})-({},{}) batch id : {}" \
                          .format(number, label, confidence, xmin, ymin, xmax, ymax, i))
                    output[i].append((xmin, ymin, xmax, ymax, confidence))

    # -----------------------------------------------------------------------------------------------------
    # --------------------------- Output images -----------------------------------------------------------
    log.info(('Sav' if args.save else 'Show') + 'ing images')
    if args.save:
        outdir = args.input[0] + '/' + 'results' + '/'
        os.makedirs(outdir, exist_ok=True)

    for i in range(len(images)):
        img = images[i]
        img = img.transpose((1, 2, 0))
        for box in output[i]:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        if args.save:
            base = os.path.basename(f"{i}.jpg")
            log.info(f'Write to {outdir + base}')
            cv2.imwrite(outdir + base, img)
        else:
            cv2.imshow('result', img)
            cv2.waitKey(0)
    # -----------------------------------------------------------------------------------------------------

    log.info("Execution successful\n")
    """
    img_array = []

    for filename in sorted(glob.glob(outdir + '*.jpg'), key=os.path.getmtime):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(path + 'project_3.avi',cv2.VideoWriter_fourcc(*'DIVX'), 28, size)

    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()"""

if __name__ == '__main__':
    sys.exit(main() or 0)