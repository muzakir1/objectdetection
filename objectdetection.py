import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Load pre-trained YOLO model
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Function for object detection
def detect_objects(uploaded_file, confidence_threshold=0.5, nms_threshold=0.3):
    # Convert BytesIO to numpy array
    image_np = np.array(Image.open(uploaded_file))

    # Get image dimensions
    height, width = image_np.shape[:2]

    # Create blob from image
    blob = cv2.dnn.blobFromImage(image_np, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Set blob as input to YOLO network
    net.setInput(blob)

    # Get output layer names
    output_layers = net.getUnconnectedOutLayersNames()

    # Perform forward pass and get output
    detections = net.forward(output_layers)

    # Process and generate detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                center_x, center_y, w, h = (obj[0:4] * np.array([width, height, width, height])).astype('int')
                x, y = int(center_x - w/2), int(center_y - h/2)
                cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)

                label = f"{classes[class_id]}: {confidence:.2f}"

                # Display class label and confidence using cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image_np, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image_np

# Streamlit application
def main():
    st.title("Deteksi Objek Sederhana dengan Streamlit")

    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Gambar yang Diunggah", use_column_width=True)

        if st.button("Deteksi Objek"):
            result_image = detect_objects(uploaded_file)

            # Display the result of object detection
            st.image(result_image, caption="Hasil Deteksi Objek", use_column_width=True, channels="BGR")

if __name__ == "__main__":
    main()
