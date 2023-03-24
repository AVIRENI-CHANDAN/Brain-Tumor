from flask import Flask, render_template, request
import numpy as np
from PIL import Image
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import load_model
from keras.utils.vis_utils import plot_model

MODEL_FILE_NAME = "model.h5"

app=Flask(__name__, template_folder="frontend", static_folder="frontend")

def format_image(image):
    return Image.open(image).resize((224, 224))

@app.route('/', methods=['GET','POST'])
def home():
    if request.method=='GET':
        return render_template("index.html")
    if request.method=='POST':
        image_file = format_image(request.files.get('file'))
        print("Image file:",image_file, image_file.size)
        image_array = img_to_array(image_file)
        print("Image array:",image_array, image_array.shape)
        img_batch = np.expand_dims(image_array, axis=0)
        img_batch /= 255
        
        
        model = load_model(MODEL_FILE_NAME, compile=True)
        print("Model:",model)
        plot_model(model, show_shapes=True, show_layer_names=True)
        confidence = np.ravel(model.predict(img_batch))[0]*100
        print("Confidence", confidence)
        response = f"Tumor detected - {round(confidence, 2)}" if confidence>=60.0 else "No tumor detected"
        print("Response", response)
        return render_template("output.html", response=response)

if __name__=="__main__":    
    app.run(debug=True)