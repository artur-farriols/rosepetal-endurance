import argparse
import os
import time

import cv2
import grpc
import endurance_pb2
import endurance_pb2_grpc
import numpy as np
from rosepetal.file_management.file_manager import FileManager

import utils.server_info as SI

def parse_args():
    parser = argparse.ArgumentParser(description="gRPC client for Endurance Service")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Predict
    predict_parser = subparsers.add_parser("predict", help="Predict using a model")
    predict_parser.add_argument("-mc", "--model_component", type=str, required=True, help="Model component")
    # predict_parser.add_argument("-w", "--width", type=int, required=True, help="Image width")
    # predict_parser.add_argument("-h", "--height", type=int, required=True, help="Image height")
    predict_parser.add_argument("-mn", "--model_name", required=True, help="Model name")
    predict_parser.add_argument("-ic", "--invert_channels", action="store_true", help="Whether to invert image channels")

    # Add Model
    add_model_parser = subparsers.add_parser("add_model", help="Add a new model")
    add_model_parser.add_argument("-mn", "--model_name", required=True, help="Model name")
    add_model_parser.add_argument("-mi", "--model_info", required=True, help="Model information as bytes")
    add_model_parser.add_argument("-mf", "--model_file", required=True, help="Model '.pt' file")

    # Change Model
    load_model_parser = subparsers.add_parser("load_model", help="Loads the specified model")
    load_model_parser.add_argument("-mn", "--model_name", required=True, help="Model name")


    # Delete Model
    delete_model_parser = subparsers.add_parser("delete_model", help="Delete an existing model")
    delete_model_parser.add_argument("-mn", "--model_name", required=True, help="Model name")

    # List Models
    list_models_parser = subparsers.add_parser("list_models", help="List all models")

    # Toggle Logger
    toggle_logger_parser = subparsers.add_parser("toggle_logger", help="Toggle logger active status")
    toggle_logger_parser.add_argument("-a", "--active", type=bool, required=True, help="Logger active status")


    return parser.parse_args()

def run():
    args = parse_args()

    # Create a gRPC channel and client
    channel = grpc.insecure_channel(f'localhost:{SI.PORT}', options=[
        ('grpc.max_send_message_length', f'{SI.MESSAGE_LENGTH}'),
        ('grpc.max_receive_message_length', f'{SI.MESSAGE_LENGTH}')
    ])
    client = endurance_pb2_grpc.EnduranceServiceStub(channel)

    if args.command == "predict":
        image_path_1 = "/opt/rosepetal-yoloseg/server/test_images/grid/1708958407381-FalsoNOK_part1.png"
        image_path_2 = "/opt/rosepetal-yoloseg/server/test_images/grid/1708958407381-FalsoNOK_part1.png"
        image_path_3 = "/opt/rosepetal-yoloseg/server/test_images/grid/1708958407381-FalsoNOK_part1.png"
        image_path_4 = "/opt/rosepetal-yoloseg/server/test_images/grid/1708958407381-FalsoNOK_part1.png"

        # Load the images
        image_1 = cv2.imread(image_path_1)
        image_2 = cv2.imread(image_path_2)
        image_3 = cv2.imread(image_path_3)
        image_4 = cv2.imread(image_path_4)

        # Since all paths are the same, you can actually load the image once and duplicate the reference
        # Concatenate images horizontally
        concatenated_image = cv2.hconcat([image_1, image_2, image_3, image_4])
        # image_shared_memory = generate_memory_handle(args.model_component, image_path)
        images = [
            create_image_shared_memory(concatenated_image, iteration=1),
            # create_image_shared_memory(image_path, iteration=2),
            # create_image_shared_memory(image_path, iteration=3),
            # create_image_shared_memory(image_path, iteration=4),
        ]

        t1 = time.perf_counter()
        # Create the PredictRequest message
        predict_request = endurance_pb2.PredictRequest(
            image = images[0],
            model_name=args.model_name,
            invert_channels=args.invert_channels,
        )

        # for image in images:

        #     predict_request.image.add().CopyFrom(image)  # Use add() and CopyFrom to append each image

        response = client.predict(predict_request)
        print(response)
        t2 = time.perf_counter()

        file_path = "/opt/rosepetal-yoloseg/server/shared_memory/request_image_1_mask"
        with open(file_path, 'rb') as file:
            byte_data = file.read()
        
        image_array = np.frombuffer(byte_data, dtype=np.uint8)
        image = image_array.reshape((400, 2200, 3))  # Assuming the image is RGB
        cv2.imwrite("/opt/rosepetal-yoloseg/server/shared_memory/mask.png", image)

        print(f"Total time: {t2 - t1}")

    if args.command == "add_model":
        model_bytes = read_file_as_bytes(args.model_file)
        model_info = read_file_as_bytes(args.model_info)
        model = endurance_pb2.PytorchModel(model=model_bytes)

        response = client.add_model(endurance_pb2.AddModelRequest(
            model_name=args.model_name,
            model_info=model_info,
            model=model,
        ))

    if args.command == "delete_model":
        response = client.delete_model(endurance_pb2.DefaultRequest(
            model_name=args.model_name
        ))

    if args.command == "list_models":
        response = client.list_models(endurance_pb2.ListModelsRequest())

        print(response)

    if args.command == "load_model":
        response = client.load_model(endurance_pb2.DefaultRequest(
            model_name=args.model_name
        ))

        print(response)

def create_image_shared_memory(image_path, model_component='default', iteration=0):
    """Simulate reading an image and creating ImageSharedMemory message."""
    # Read an image (dummy operation)
    file_path = f"/opt/rosepetal-yoloseg/server/shared_memory/request_image_{iteration}"
    name = os.path.basename(file_path)
    mask_name = name + "_mask"
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
    else:
        image = image_path
    height, width, channels = image.shape
    print(height, width, channels)

    # Convert the image to bytes (for demonstration, not using actual shared memory)
    image_bytes = image.tobytes()

    with open(file_path, 'wb') as file:
        file.write(image_bytes)

    # Create a Bitmap message
    bitmap = endurance_pb2.Bitmap(
        width=width,
        height=height,
        shared_memory_handle=endurance_pb2.SharedMemoryHandle(
            name=name,  # This should be the actual shared memory identifier
            size=str(len(image_bytes)),
            offset='0'
        )
    )

    # Create an AnomalyMaskParams message (dummy, adjust as needed)
    anomaly_mask_params = endurance_pb2.AnomalyMaskParams(
        shared_memory_handle=endurance_pb2.SharedMemoryHandle(
            name=mask_name,
            size='0',
            offset='0'
        )
    )

    # Create and return an ImageSharedMemory message
    return endurance_pb2.ImageSharedMemory(
        model_component=model_component,
        bitmap=bitmap,
        anomaly_mask_params=anomaly_mask_params
    )


# def generate_memory_handle(model_component, image_path):
#     image = cv2.imread(image_path)
#     file_path = "/opt/rosepetal-yoloseg/server/shared_memory/request_image"
#     height, width = image.shape[:2]

#     t_bytes_start = time.perf_counter()
#     byte_data = image.tobytes()
#     t_bytes_end = time.perf_counter()

#     print(f"Elapsed to bytes  time is: {t_bytes_end - t_bytes_start}")

#     t_save_start = time.perf_counter()
#     with open(file_path, 'wb') as file:
#         file.write(byte_data)
#     t_save_end = time.perf_counter()

#     print(f"Elapsed saving  time is: {t_save_end - t_save_start}")

#     shared_memory_handle = endurance_pb2.SharedMemoryHandle(
#         name="request_image",
#         size=str(len(byte_data)),
#         offset="0"
#     )
#     bitmap = endurance_pb2.Bitmap(
#         width=width,
#         height=height,
#         shared_memory_handle=shared_memory_handle
#     )
#     anomaly_mask_params = endurance_pb2.AnomalyMaskParams(
#         shared_memory_handle=shared_memory_handle
#     )
#     image_shared_memory = endurance_pb2.ImageSharedMemory(
#         model_component=model_component,
#         bitmap=bitmap,
#         anomaly_mask_params=anomaly_mask_params
#     )

#     return image_shared_memory

def read_file_as_bytes(file_path):
    """Read and return the content of a file as bytes."""
    with open(file_path, 'rb') as file:
        return file.read()


if __name__ == "__main__":
    run()