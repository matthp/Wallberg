########################################
# CLIP
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModelForObjectDetection
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from torchvision.transforms import functional as F

device = "cuda:0"

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IOU) between two bounding boxes.

    Parameters:
    - box1, box2: Lists representing the coordinates of the top-left and bottom-right corners of the boxes.
      Format: [x1, y1, x2, y2]

    Returns:
    - iou: Intersection over Union value.
    """
    # Calculate the intersection area
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    intersection_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)

    # Calculate the union area
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    union_area = box1_area + box2_area - intersection_area

    # Calculate IOU
    iou = intersection_area / union_area

    return iou

def non_maximum_suppression(boxes, scores, threshold=0.0):
    """
    Perform Non-Maximum Suppression (NMS) on a list of bounding boxes with corresponding scores.

    Parameters:
    - boxes: List of boxes where each box is represented as [x1, y1, x2, y2].
    - scores: List of scores corresponding to each box.
    - threshold: IOU threshold to determine overlapping boxes.

    Returns:
    - selected_boxes: List of boxes selected after NMS.
    """
    # Sort boxes based on scores (in descending order)
    sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)

    selected_boxes = []

    while len(sorted_indices) > 0:
        current_index = sorted_indices[0]
        selected_boxes.append(boxes[current_index])

        # Remove the current box from consideration
        del sorted_indices[0]

        # Filter out boxes with high IOU
        sorted_indices = [i for i in sorted_indices if calculate_iou(boxes[current_index], boxes[i]) <= threshold]

    return selected_boxes


# Function to extract bounding box regions as new images with resizing
from torchvision.transforms import functional as F
from PIL import Image

# Assuming `image` is a PIL Image

# Function to extract bounding box regions as new images
def extract_bounding_box_regions(image, boxes):
    # Loop through each bounding box and extract the region as a new image
    extracted_images = []
    for box in boxes:
        box = [int(i) for i in box]

        # Extract region using PIL
        region_image = image.crop(box)

        # Append the extracted image tensor to the list
        extracted_images.append(region_image)

    return extracted_images

def resize_images(image_list, target_size=(640, 640)):
    resized_images = []

    for img in image_list:
        # Calculate aspect ratio
        width, height = img.size
        aspect_ratio = width / height

        # Calculate new dimensions to fit target size while preserving aspect ratio
        if aspect_ratio > 1:
            new_width = target_size[0]
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = target_size[1]
            new_width = int(new_height * aspect_ratio)

        # Resize image with black bars
        resized_img = F.resize(img, (new_width, new_height))
        padding_left = (target_size[0] - new_width) // 2
        padding_top = (target_size[1] - new_height) // 2
        padded_img = Image.new("RGB", target_size, (0, 0, 0))
        padded_img.paste(resized_img, (padding_left, padding_top))

        resized_images.append(padded_img)

    return resized_images

class Detector:
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32"):
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)

        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True).to(device)

        self.device = device

    def detect_person_bounding_boxes(self, image, detection_threshold=0.1):

        with torch.no_grad():
            results = self.yolo_model(image)
            results = results.xyxy[0]

            if results.shape[0] == 0:
                return [], [], []
            else:
                results = results[results[::, -1] == 0, ::]
                results = results[results[::, -2] >= detection_threshold, ::]
                scores = results[::, -2].detach().cpu().numpy().tolist()
                boxes = results[::, 0:4].detach().cpu().numpy().tolist()

        # boxes = non_maximum_suppression(boxes, scores)

        extracted_images = resize_images(extract_bounding_box_regions(image, boxes))
        return boxes, scores


class Tracker:
    def __init__(self, max_frames_to_persist=40):
        self.max_frames_to_persist = max_frames_to_persist
        self.objects = {}  # Dictionary to store tracked objects {object_id: [last_known_position, frames_persisted, ]}

        self.detector = Detector()

    def update(self, frame):
        # Get current bounding boxes from the frame
        detected_boxes, _ = self.detector.detect_person_bounding_boxes(frame)

        # Update existing tracked objects
        updated_objects = {}
        associated_ids = []

        # Calculate IOU between all pairs of tracked and detected boxes
        iou_matrix = np.zeros((len(self.objects), len(detected_boxes)))
        for i, (object_id, (last_known_position, _, _)) in enumerate(self.objects.items()):
            for j, box in enumerate(detected_boxes):
                iou_matrix[i, j] = calculate_iou(last_known_position, box)

        # Solve the assignment problem using the Hungarian algorithm
        tracked_indices, detected_indices = linear_sum_assignment(-iou_matrix)

        for tracked_index, detected_index in zip(tracked_indices, detected_indices):
            object_id = list(self.objects.keys())[tracked_index]
            matched_iou = iou_matrix[tracked_index, detected_index]

            # Check if the matched IOU is greater than 0
            if matched_iou > 0:
                matched_box = detected_boxes[detected_index]
                updated_objects[object_id] = [matched_box, 0]
                associated_ids.append(object_id)

        # Add new objects
        new_objects = {}
        new_ids = []

        for idx, box in enumerate(detected_boxes):
            if all(calculate_iou(last_known_position, box) == 0 for last_known_position, _, _ in updated_objects.values()):
                new_object_id = max(self.objects.keys(), default=0) + 1
                new_objects[new_object_id] = [box, 0]
                new_ids.append(new_object_id)

        # Combine updated and new objects
        self.objects = {**updated_objects, **new_objects}

        # Update persisted frames for unassociated tracked objects
        for object_id, (last_known_position, frames_persisted, s_p) in self.objects.items():
            if object_id not in associated_ids:
                self.objects[object_id] = [last_known_position, frames_persisted + 1, s_p]

        # Remove objects that exceed the maximum frames to persist
        self.objects = {object_id: [position, frames, s_p] for object_id, (position, frames, s_p) in self.objects.items()
                        if frames <= self.max_frames_to_persist}

        # Prepare the output tuple
        last_known_positions = [position for position, _, _ in self.objects.values()]
        all_ids = associated_ids + new_ids
        s_p = [x for _, _, x in self.objects.values()]

        return last_known_positions, all_ids, s_p