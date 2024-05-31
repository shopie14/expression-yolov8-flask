from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from skimage import exposure
from ultralytics import YOLO
import os

app = Flask(__name__)
UPLOAD_FOLDER = './static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load YOLO model
model = YOLO("Yolov8l.pt")
class_names = model.names

# Function to enhance dark images
def enhance_image(image):
    mean_brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,2])
    if mean_brightness < 100:
        image = exposure.rescale_intensity(image, in_range='image', out_range=(0, 255))
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image

# Function to identify expression from image and draw bounding boxes
def identify_expression(image, model):
    results = model.predict(image, conf=0.25)
    if results:
        for box in results[0].boxes:
            expression = class_names[int(box.cls)]
            confidence = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract bounding box coordinates
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Put label
            label = f"{expression}: {confidence:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return image, expression, confidence
    else:
        return image, None, 0.0

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            image = cv2.imread(filename)

            # Check brightness and enhance if necessary
            mean_brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,2])
            enhanced_image = None
            enhancement_needed = False
            if mean_brightness < 100:
                enhanced_image = enhance_image(image.copy())
                enhancement_needed = True

            # Identify expression
            image_to_use = enhanced_image if enhanced_image is not None else image
            processed_image, expression, confidence = identify_expression(image_to_use, model)
            processed_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + file.filename)
            cv2.imwrite(processed_filename, processed_image)

            return render_template(
                'index.html',
                original_image=url_for('static', filename='uploads/' + file.filename),
                enhanced_image=url_for('static', filename='uploads/' + file.filename) if enhancement_needed else None,
                processed_image=url_for('static', filename='uploads/processed_' + file.filename),
                expression=expression,
                confidence=confidence
            )
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False)
