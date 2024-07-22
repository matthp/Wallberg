import cv2
from Detector import Tracker
from PIL import Image
from ArduinoCommandSender import Sender, PIDController

# Instantiate object tracker
detector = Tracker()

# Instantiate raspberry pi command server
sender_ip = "192.168.1.167"
receiver_ip = "192.168.1.183"
port = 12345

sender = Sender(sender_ip, receiver_ip, port)

# Instantiate the PID controller
pid = PIDController(0.001, 0, 0)

# Open a connection to the selected webcam
selected_camera_index = 0
cap = cv2.VideoCapture(selected_camera_index)

# Set the frame rate to 10 frames per second
cap.set(cv2.CAP_PROP_FPS, 10)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open selected webcam.")
    exit()

# Create a window to display the webcam feed (optional)
cv2.namedWindow("Webcam Feed", cv2.WINDOW_NORMAL)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Resize the frame
    resized_frame = cv2.resize(frame, (640, 640))

    # Convert BGR to RGB
    # rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    pil_image = Image.fromarray(resized_frame)

    # boxes, scores = detector.detect_person_bounding_boxes(pil_image)
    boxes, IDs = detector.update(pil_image)

    if len(boxes) > 0:
        set_point = [640//2, 640//2]
        _, pan_command = pid.compute(set_point, boxes, 0.1)
        sender.send_command(0, pan_command)


    for box, ID in zip(boxes, IDs, surrender_probs):
        box = [round(i, 2) for i in box]
        # print(f"Detected with confidence {round(score.item(), 3)} at location {box}")

        # Draw bounding box on the frame
            color = (0, 0, 255)  # Green color for the bounding box
        thickness = 10

        start_point = (int(box[0]), int(box[1]))  # Top-left corner
        end_point = (int(box[2]), int(box[3]))  # Bottom-right corner

        cv2.rectangle(resized_frame, start_point, end_point, color, thickness)

        # Display ID in the upper left corner of the bounding box
        cv2.putText(resized_frame, str(ID), (start_point[0], start_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)

        # cv2.putText(resized_frame, str(surrender_prob), (end_point[0], end_point[1] - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 10)

    # Check if the frame was read successfully
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Display the frame (optional)
    cv2.imshow("Webcam Feed", resized_frame)

    # Wait for 100 milliseconds and check for the 'q' key to exit
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# Release the webcam and close the window (optional)
cap.release()
cv2.destroyAllWindows()
