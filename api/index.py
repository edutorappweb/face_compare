import numpy as np
from flask import Flask, request, jsonify
import face_recognition
import cv2

app = Flask(__name__)

def find_face_encodings(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(rgb_image)
    return face_encodings

@app.route('/compare_faces', methods=['POST'])
def compare_faces():
    # Check if files are present in the request
    if 'image_1' not in request.files or 'image_2' not in request.files:
        return jsonify({"error": "Please provide both image_1 and image_2."}), 400

    image_1 = request.files['image_1'].read()
    image_2 = request.files['image_2'].read()

    image_1_array = cv2.imdecode(np.frombuffer(image_1, np.uint8), cv2.IMREAD_COLOR)
    image_2_array = cv2.imdecode(np.frombuffer(image_2, np.uint8), cv2.IMREAD_COLOR)

    face_encodings_1 = find_face_encodings(image_1_array)
    face_encodings_2 = find_face_encodings(image_2_array)

    if len(face_encodings_1) == 0 or len(face_encodings_2) == 0:
        return jsonify({"error": "No faces found in one or both of the images."}), 400

    # Compare each pair of face encodings
    result = {"matches": []}
    threshold = 0.6

    for encoding_1 in face_encodings_1:
        for encoding_2 in face_encodings_2:
            # Compute the Euclidean distance between the face encodings
            distance = face_recognition.face_distance([encoding_1], encoding_2)[0]

            # Check if the distance is below the threshold
            if distance <= threshold:
                accuracy = round((1 - distance) * 100, 2)
                result["matches"].append({"result": "Faces are the same.", "accuracy": accuracy})
            else:
                result["matches"].append({"result": "Faces are not the same."})

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

import numpy as np
from flask import Flask, request, jsonify
import face_recognition
import cv2

app = Flask(__name__)

def find_face_encodings(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(rgb_image)
    return face_encodings

@app.route('/compare_faces', methods=['POST'])
def compare_faces():
    # Check if files are present in the request
    if 'image_1' not in request.files or 'image_2' not in request.files:
        return jsonify({"error": "Please provide both image_1 and image_2."}), 400

    image_1 = request.files['image_1'].read()
    image_2 = request.files['image_2'].read()

    image_1_array = cv2.imdecode(np.frombuffer(image_1, np.uint8), cv2.IMREAD_COLOR)
    image_2_array = cv2.imdecode(np.frombuffer(image_2, np.uint8), cv2.IMREAD_COLOR)

    face_encodings_1 = find_face_encodings(image_1_array)
    face_encodings_2 = find_face_encodings(image_2_array)

    if len(face_encodings_1) == 0 or len(face_encodings_2) == 0:
        return jsonify({"error": "No faces found in one or both of the images."}), 400

    # Compare each pair of face encodings
    matches = []

    for encoding_1 in face_encodings_1:
        for encoding_2 in face_encodings_2:
            # Compute the Euclidean distance between the face encodings
            distance = face_recognition.face_distance([encoding_1], encoding_2)[0]

            # Check if the distance is below the threshold
            if distance <= threshold:
                matches.append("Faces are the same.")
            else:
                matches.append("Faces are not the same.")

    return jsonify(matches)

if __name__ == '__main__':
    app.run(debug=True)
