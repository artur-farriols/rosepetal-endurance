import logging
from concurrent import futures
import os
import time

import cv2 
import grpc
import endurance_pb2
import endurance_pb2_grpc
import numpy as np
from rosepetal.file_management.file_manager import FileManager
from ultralytics import YOLO

import utils.server_info as SI
from utils.copy_model_files import copy_model_files
from utils.logger import get_logger

logger = get_logger()

class EnduranceService(endurance_pb2_grpc.EnduranceServiceServicer):
    """A gRPC service handler for Endurance segmentation service."""

################################################################# 
# CONSTRUCTOR
################################################################# 

    def __init__(self):
        """Initializes the EnduranceService."""

        self.model_name = None
        self.model = None
        self.min_score_map = None

        self.path = os.path.dirname(os.path.realpath(__file__))

        self.width = None
        self.height = None
        self.channels = None

        self.class_mapping = None


################################################################# 
# MAIN FUNCTIONS
################################################################# 

    def predict(self, request, context):
        t_total_start = time.perf_counter()
        logger.info("predict request")

        if self.model is None and request.model_name == "":
            raise ValueError("No model is loaded and no model name was provided in the request.")
        
        # Contains the images sent from the Node-RED client
        batch_images = []
        # Contains the images names 
        batch_names = []
        # Contains the mapping between the input names and output
        batch_results_info = {}
        
        image_shared_memory = request.image
        print(image_shared_memory)

        # Process each ImageSharedMemory object
        bitmap = image_shared_memory.bitmap
        width = bitmap.width
        height = bitmap.height
        shared_memory_handle = bitmap.shared_memory_handle
        shared_memory_name = shared_memory_handle.name
        anomaly_mask_params = image_shared_memory.anomaly_mask_params
        
        # Decode the image
        t_decode_start = time.perf_counter()
        image_array = np.frombuffer(self._read_bytes(os.path.join(self.path, SI.SHARED_MEMORY_PATH(shared_memory_name=shared_memory_name))), dtype=np.uint8)
        image = image_array.reshape((height, width, 3))  # Assuming the image is RGB
        t_decode_end = time.perf_counter()

        logger.info(f"Decoding image took: {t_decode_end - t_decode_start} seconds")
        
        # Add the image to the batch
        batch_images = self._divide_image(image)

        min_score_map = request.scores
        print(min_score_map)
        if min_score_map:
            self.min_score_map = min_score_map
        elif self.min_score_map is None:
            self.min_score_map = SI.DEFAULT_MIN_SCORE_MAP


        # batch_images.append(image)
        # batch_names.append(shared_memory_name)
        # batch_results_info[shared_memory_name] = anomaly_mask_params

        print(len(batch_images))
        print(np.shape(batch_images))
        print(f"Decoding image: {t_decode_end - t_decode_start}")

        # if np.shape():

        if self.model is None:
            self._start_model(request.model_name)

        results = self.model.predict(batch_images)
        # print(dir(results[0][0]))
        print("HERE")
        response = self._process_results(results, shared_memory_name, anomaly_mask_params)

        t_total_end = time.perf_counter()

        print(f"Total time: {t_total_end - t_total_start}")

        # Constructing and returning the PredictResponse
        return response   
    
    def add_model(self, request, context):
        logger.info("add_model request")

        model_name = request.model_name
        model_info = request.model_info
        model = request.model
        model_path = FileManager.merge_paths(self.path, SI.MODEL_PATH(model_name))
        model_files = self._get_model_files(model)
        model_files[SI.MODEL_CONFIG] = model_info
        
        copy_model_files(model_path=model_path, model_files=model_files)
        
        logger.info(f"Model {model_name} loaded successfully.")
        return endurance_pb2.DefaultResponse(done=True)
    
    def load_model(self, request, context):
        model_name = request.model_name
        min_score_map = request.min_score_map

        self._start_model(model_name)

        if min_score_map:
            self.min_score_map = min_score_map
        elif self.min_score_map is None:
            self.min_score_map = SI.DEFAULT_MIN_SCORE_MAP

        return endurance_pb2.DefaultResponse(done=True)


################################################################# 
# AUXILIAR FUNCTIONS
#################################################################    

    def _read_bytes(self, file_path: str):
        print(file_path)
        with open(file_path, 'rb') as file:
            byte_data = file.read()
        return byte_data
    
    def _get_model_files(self, model):
        model_files = {SI.PYTORCH_MODELS[0]: model.model}

        return model_files
    
    def _start_model(self, model_name):
        model_path = FileManager.merge_paths(self.path, SI.MODEL_PATH(model_name=model_name))
        model_info_path = FileManager.merge_paths(model_path, SI.MODEL_CONFIG) 
        model_dimensions = FileManager.read(model_info_path)['model_config']['image_dimensions']
        class_mapping = FileManager.read(model_info_path)['model_config']['classes']
        print(model_dimensions)

        yolo_path = FileManager.search(model_path, *[SI.PYTORCH_MODELS[0]])[SI.PYTORCH_MODELS[0]][0]

        self.model = YOLO(model=yolo_path)
        self.width = model_dimensions['width']
        self.height = model_dimensions['height']
        self.channels = model_dimensions['channels']
        self.class_mapping = class_mapping

    def _process_results(self, results, name, anomaly_mask_params):
        t1 = time.perf_counter()
        prediction = endurance_pb2.Prediction(name=name)
        print('HEY')
        # print(dir(results[0]))
        # print(results[0])
        classes_dict = results[0].names
        print('CCC')
        print('BBB', results[0].orig_shape)
        original_height, original_width = results[0].orig_shape
        compressed_masks = []
        print(original_height, original_width)

        for counter, result in enumerate(results):

            # print(dir(result))
            # print(dir(result.boxes))
            # print(result.boxes)
            print(dir(result.masks))
            print(result.masks)

            masks = result.masks.data.cpu().numpy() if result.masks is not None else None

            if masks is not None:
                print("YEPA")
                print(len(result.masks))
                print(result.masks.shape)
                compressed_mask = np.zeros((result.masks.shape[1:]),dtype=np.uint8)
                compressed_mask = np.stack([compressed_mask, compressed_mask, compressed_mask], axis=-1)

                predictions = result.boxes.data.cpu().numpy()

                filtered_predictions = []
                filtered_predictions, filtered_masks = self._filter_predictions(predictions, masks)
                # for prediction, mask in zip(predictions, masks):

                bounding_boxes = [row[:4] for row in predictions]
                bounding_boxes = [[round(element) for element in row] for row in bounding_boxes]
                scores = [row[4] for row in predictions]
                classes = [row[5] for row in predictions]
                classes_str = [classes_dict[number] for number in classes]
                print(classes_str)
    
                for j, bounding_box in enumerate(bounding_boxes):
                    x1, y1, x2, y2 = bounding_box
                    y1 += counter * 550
                    x2 += counter * 550
                    y2 += counter * 550
                    score = scores[j]
                    class_name = classes_str[j]

                    box = endurance_pb2.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                    prediction.boxes.append(box)
                    prediction.classes.append(class_name)
                    prediction.scores.append(score)

                for mask, class_number in zip(masks, classes):
                    # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                    compressed_mask[mask == 0] = SI.BACKGROUND_COLOR
                    compressed_mask[mask == 1] = SI.COLOR_PALETTE[class_number]
                
                compressed_mask = cv2.resize(compressed_mask, (original_width, original_height))
                
                # compressed_mask = cv2.cvtColor(compressed_mask, cv2.COLOR_GRAY2RGB)
                # compressed_masks.append(compressed_mask)        
                
            else:    
                print("YEPINS")
                compressed_mask = np.ones((original_height, original_width),dtype=np.uint8) * 255
                compressed_mask = np.stack([compressed_mask, compressed_mask, compressed_mask], axis=-1)
            
            compressed_masks.append(compressed_mask)

        processed_results = endurance_pb2.PredictResponse()
        processed_results.prediction.CopyFrom(prediction)

        # processed_results.prediction.add().CopyFrom(prediction)

        final_mask = cv2.hconcat(compressed_masks)
        final_mask_bytes = final_mask.tobytes()
        print("anomaly_mask_params.shared_memory_handle.name", anomaly_mask_params.shared_memory_handle.name)
        file_path = FileManager.merge_paths(self.path, SI.SHARED_MEMORY_PATH(anomaly_mask_params.shared_memory_handle.name))
        with open(file_path, 'wb') as file:
            file.write(final_mask_bytes)

        t2 = time.perf_counter()
        print(f"Results processed in: {t2 - t1}")
        print(classes_dict)

        return processed_results
    
    def _filter_predictions(self, predictions, masks):
        filtered_predictions, filtered_masks = self._filter_predictions_by_score(predictions, masks)
        filtered_predictions, filtered_masks = self._filter_predictions_by_area_and_margin_proximity(predictions, masks)

        return filtered_predictions, filtered_masks


    def _filter_predictions_by_score(self, predictions, masks):
        filtered_predictions = [] 
        filtered_masks = []
        for prediction, mask in zip(predictions, masks):
            predicted_class = prediction[5]
            predicted_class_str = self.class_mapping[str(int(predicted_class))]
            if prediction[4] > self.min_score_map[predicted_class_str]:
                filtered_predictions.append(prediction)
                filtered_masks.append(mask)
            print(prediction)

        return np.array(filtered_predictions), np.array(filtered_masks)

    def _filter_predictions_by_area_and_margin_proximity(self, predictions, masks):
        return predictions, masks
    
    
    def _divide_image(self, image):
        # Calculate the width of each column
        colWidth = image.shape[1] // 4
        
        # Initialize an empty list to store the sub-images (columns)
        subImages = []
        
        # Loop through four times to extract each column
        for i in range(4):
            # Calculate the starting and ending x-coordinate for each column
            startX = i * colWidth
            endX = startX + colWidth
            
            # Use slicing to extract the column
            # Note: image[rows, cols], where ':' means all rows, and startX:endX specifies the column range
            subImage = image[:, startX:endX]
            
            # Append the sub-image (column) to the list
            subImages.append(subImage)
        
        # Return the list of sub-images
        return subImages


################################################################# 
# SERVER FUNCTIONS
################################################################# 

def serve():
    options = [
        ('grpc.max_receive_message_length', SI.MESSAGE_LENGTH),
        ('grpc.max_send_message_length', SI.MESSAGE_LENGTH)
    ]

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    endurance_pb2_grpc.add_EnduranceServiceServicer_to_server(EnduranceService(), server)
    server.add_insecure_port(f'[::]:{SI.PORT}')
    server.start()
    logger.info(f"server listening on port: {SI.PORT}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        # Stop the server gracefully on Ctrl+C
        server.stop(0)
        logger.info("Server execution stopped by user")
    except Exception as e:
        server.stop(0)
        logger.error(f"The following error has interrupted the server execution: {e}")

if __name__ == '__main__':
    if SI.DEBUG:
        os.environ['GRPC_VERBOSITY'] = 'debug'
    serve()