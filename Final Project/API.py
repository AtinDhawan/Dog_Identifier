from fastapi import FastAPI, UploadFile, File, Form
import numpy as np
import base64
from functions import *
from tensorflow import keras
# from registry import load_Xception_model, load_YOLO_model




app = FastAPI()

# Loading in models:
Xception_model = load_Xception_model()
model = load_YOLO_model()

@app.post('/predict')
def file_upload(
        my_file: bytes = File(...),
        shape: str = Form(...),
        dtype: str = Form(...)):
    from_bytes = np.frombuffer(my_file, dtype = dtype)
    reshape = from_bytes.reshape(eval(shape))
    # convert to tensor
    Xception_model = keras.models.load_model('Xception_model.h5')
    model = load_YOLO_model()
    print(reshape.shape)
    breed_list = return_breeds(model, reshape)[1]
    response = str(breed_list)
    #results = make_predictions(list_images, app.state.cnn_model)
    return {"response": response}
