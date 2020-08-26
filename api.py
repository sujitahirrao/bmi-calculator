import os
from datetime import datetime
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

from src import config
from src.predict_bmi_from_face_image import predict_from_face_image
from src.predict_bmi_from_med_doc import predict_from_med_doc

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask("BMI Calculator")
app.config["UPLOAD_FOLDER"] = config.DATA_FOLDER
router = '/bmi-calculator/' + config.ENV


def allowed_file(file_name):
    return '.' in file_name and \
           file_name.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_file(request_id, request_file_part):
    # check if POST request has the file part
    if "image_file" not in request_file_part:
        msg = "No file part: 'image_file' in POST request."
        print(msg)
        raise Exception(msg)
    file = request_file_part["image_file"]
    if file.filename == '':
        msg = "File name is empty."
        print(msg)
        raise Exception(msg)
    if file and allowed_file(file.filename):
        print("Uploaded file:\t", file.filename)
        image_file_name = str(datetime.now()).replace(':', '') + \
                          f'_{request_id}_' + secure_filename(file.filename)
        image_file_path = os.path.join(
            app.config["UPLOAD_FOLDER"], image_file_name)
        file.save(image_file_path)
        print(f"Image file is saved at {image_file_path}")
        return image_file_path


@app.route(router + '/hello', methods=['GET'])
def hello():
    return "Hello from BMI Calculator!"


@app.route(router + '/predict-bmi-from-face-image',
           methods=['GET', 'POST'])
def _predict_from_face_image():
    try:
        if request.method == 'POST':
            print()
            print("request.form:\t", request.form)
            print("request.files:\t", request.files)
            request_id = request.form.get("request_id", 100)
            print("request_id:\t", request_id)
            image_file_path = save_file(request_id, request.files)
            if image_file_path:
                bmi = predict_from_face_image.predict(image_file_path)
                print("BMI:\t", bmi)
                return jsonify({"request_id": request_id, "bmi": float(bmi)})
            else:
                return "Failed to save image file."
        else:
            msg = "POST the image file. Don't GET it."
            print(msg)
            return msg
    except Exception as e:
        msg = str(type(e).__name__) + ': ' + str(e)
        print(msg)
        return msg


@app.route(router + '/predict-bmi-from-med-doc',
           methods=['GET', 'POST'])
def _predict_from_med_doc():
    try:
        if request.method == 'POST':
            print()
            print("request.form:\t", request.form)
            print("request.files:\t", request.files)
            request_id = request.form.get("request_id", config.DEFAULT_REQUEST_ID)
            print("request_id:\t", request_id)
            image_file_path = save_file(request_id, request.files)
            if image_file_path:
                bmi = predict_from_med_doc.predict(image_file_path)
                print("BMI:\t", bmi)
                return jsonify({"request_id": request_id, "bmi": float(bmi)})
            else:
                return "Failed to save image file."
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
