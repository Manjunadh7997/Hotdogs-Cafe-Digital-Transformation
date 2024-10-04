from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded_images/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load your trained model
model = load_model(r'C:/work_files/projects/Hot_dogs/HOTDOGS_CLASSIFICATION/model/Hot_dogs_CNN.h5')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def prepare_image(file_path):
    img = image.load_img(file_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():    
    print(request.method)
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            print(file_path)
            
            img_array = prepare_image(file_path)
            print(img_array)
            prediction = model.predict(img_array)
            print(prediction)
            label = 'Hotdog Enjoy the hotdog in each bite and taste few more in hotdogs. ' if prediction[0][0] < 0.5 else 'Not a Hotdog, click on predict to upload Images '
            
            return render_template('result.html', label=label, image_url=file_path)
    return render_template('predict.html')

@app.route('/submit')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
