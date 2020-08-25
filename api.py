import os
from datetime import datetime
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

import config
from detect_and_calculate_bmi import detect_and_calculate

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask("BMI Calculator")
app.config["UPLOAD_FOLDER"] = config.DATA_FOLDER


def allowed_file(file_name):
    return '.' in file_name and \
           file_name.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/bmi-calculator/hello', methods=['GET'])
def home():
    return "Hello from BMI Calculator!"


@app.route('/bmi-calculator/calculate-bmi-from-face-image',
           methods=['GET', 'POST'])
def calculate_from_face_image():
    try:
        if request.method == 'POST':
            print()
            print("request.form:\t", request.form)
            print("request.files:\t", request.files)
            request_id = request.form["request_id"]
            print("request_id:\t", request_id)
            if "image_file" not in request.files:
                msg = "No file part: 'image_file' in POST request."
                print(msg)
                return msg
            file = request.files["image_file"]
            if file.filename == '':
                msg = "File name is empty."
                print(msg)
                return msg
            if file and allowed_file(file.filename):
                print("Uploaded file:\t", file.filename)
                image_file_name = str(datetime.now()).replace(':', '') + \
                                  f'_{request_id}_' + secure_filename(file.filename)
                image_file_path = os.path.join(app.config["UPLOAD_FOLDER"], image_file_name)
                file.save(image_file_path)
                print(f"Image file is saved at {image_file_path}")
                bmi = detect_and_calculate(image_file_path)
                print("BMI:\t", bmi)
                return jsonify({"request_id": request_id, "bmi": float(bmi)})
        else:
            msg = "POST the image file. Don't GET it."
            print(msg)
            return msg
    except Exception as e:
        msg = str(type(e).__name__) + ': ' + str(e)
        print(msg)
        return msg


if __name__ == '__main__':
    app.run(debug=False)
