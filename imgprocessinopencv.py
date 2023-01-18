from flask import Flask, jsonify, request
import cv2
from skimage import io
from skimage.transform import resize
from torchvision import models

app = Flask(__name__)

# Load the pre-trained deep learning model
model = models.resnet18(pretrained=True)
model.eval()

@app.route('/image-processing', methods=['POST'])
def image_processing():
    # Get the image file from the request
    image_file = request.files['image']
    # Open the image using OpenCV
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    # Perform the image processing using OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    # Perform the image processing using scikit-image
    image = resize(image, (64, 64))
    image = image.astype('float32') / 255
    # Perform the image

    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)

    predictions = model(image)
    predictions = predictions.argmax()
    return jsonify({"prediction": predictions})
    
if name == 'main':
app.run(debug=True)