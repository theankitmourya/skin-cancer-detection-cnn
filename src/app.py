from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import os
# Load the model
from keras.models import load_model


Model = load_model('skincancerdetectionmodel.h5')

lesion_classes_dict = {
    0: 'Melanocytic nevi',
    1: 'Melanoma-Cancerous',
    2: 'Benign keratosis-like lesions',
    3: 'Basal cell carcinoma-cancerous',
    4: 'Actinic keratoses',
    5: 'vascular lesions',
    6: 'Dermatofibroma'
}

def model_predict(img_path, Model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) 
    x = x / 255.0
    preds = Model.predict(x)
    return preds

def browse_image():
    global file_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
    if file_path:
        display_image(file_path)

def predict():
    if file_path:
        preds = model_predict(file_path, Model)
        pred_class = preds.argmax(axis=1)
        pr = lesion_classes_dict[pred_class[0]]
        result_label.config(text="Predicted class: " + pr)
        print("Predicted class index:", pred_class[0]) 

def display_image(file_path):
    img = Image.open(file_path)
    img.thumbnail((300, 300))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

window = Tk()
window.title("Skin Cancer Detection")
window.geometry("400x400")

browse_button = Button(window, text="Browse Image", command=browse_image)
image_label = Label(window)
predict_button = Button(window, text="Predict", command=predict)
result_label = Label(window, text="")

browse_button.pack(pady=10)
image_label.pack()
predict_button.pack(pady=10) 
result_label.pack(pady=10)

window.mainloop()
