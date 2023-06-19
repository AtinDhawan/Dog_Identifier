from ultralytics import YOLO
import pandas as pd
import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from tensorflow import keras
from keras.applications.resnet_v2 import preprocess_input, decode_predictions

# Dog names
dog_names = ['n/001.Affenpinscher',
 'n/002.Afghan_hound',
 'n/003.Airedale_terrier',
 'n/004.Akita',
 'n/005.Alaskan_malamute',
 'n/006.American_eskimo_dog',
 'n/007.American_foxhound',
 'n/008.American_staffordshire_terrier',
 'n/009.American_water_spaniel',
 'n/010.Anatolian_shepherd_dog',
 'n/011.Australian_cattle_dog',
 'n/012.Australian_shepherd',
 'n/013.Australian_terrier',
 'n/014.Basenji',
 'n/015.Basset_hound',
 'n/016.Beagle',
 'n/017.Bearded_collie',
 'n/018.Beauceron',
 'n/019.Bedlington_terrier',
 'n/020.Belgian_malinois',
 'n/021.Belgian_sheepdog',
 'n/022.Belgian_tervuren',
 'n/023.Bernese_mountain_dog',
 'n/024.Bichon_frise',
 'n/025.Black_and_tan_coonhound',
 'n/026.Black_russian_terrier',
 'n/027.Bloodhound',
 'n/028.Bluetick_coonhound',
 'n/029.Border_collie',
 'n/030.Border_terrier',
 'n/031.Borzoi',
 'n/032.Boston_terrier',
 'n/033.Bouvier_des_flandres',
 'n/034.Boxer',
 'n/035.Boykin_spaniel',
 'n/036.Briard',
 'n/037.Brittany',
 'n/038.Brussels_griffon',
 'n/039.Bull_terrier',
 'n/040.Bulldog',
 'n/041.Bullmastiff',
 'n/042.Cairn_terrier',
 'n/043.Canaan_dog',
 'n/044.Cane_corso',
 'n/045.Cardigan_welsh_corgi',
 'n/046.Cavalier_king_charles_spaniel',
 'n/047.Chesapeake_bay_retriever',
 'n/048.Chihuahua',
 'n/049.Chinese_crested',
 'n/050.Chinese_shar-pei',
 'n/051.Chow_chow',
 'n/052.Clumber_spaniel',
 'n/053.Cocker_spaniel',
 'n/054.Collie',
 'n/055.Curly-coated_retriever',
 'n/056.Dachshund',
 'n/057.Dalmatian',
 'n/058.Dandie_dinmont_terrier',
 'n/059.Doberman_pinscher',
 'n/060.Dogue_de_bordeaux',
 'n/061.English_cocker_spaniel',
 'n/062.English_setter',
 'n/063.English_springer_spaniel',
 'n/064.English_toy_spaniel',
 'n/065.Entlebucher_mountain_dog',
 'n/066.Field_spaniel',
 'n/067.Finnish_spitz',
 'n/068.Flat-coated_retriever',
 'n/069.French_bulldog',
 'n/070.German_pinscher',
 'n/071.German_shepherd_dog',
 'n/072.German_shorthaired_pointer',
 'n/073.German_wirehaired_pointer',
 'n/074.Giant_schnauzer',
 'n/075.Glen_of_imaal_terrier',
 'n/076.Golden_retriever',
 'n/077.Gordon_setter',
 'n/078.Great_dane',
 'n/079.Great_pyrenees',
 'n/080.Greater_swiss_mountain_dog',
 'n/081.Greyhound',
 'n/082.Havanese',
 'n/083.Ibizan_hound',
 'n/084.Icelandic_sheepdog',
 'n/085.Irish_red_and_white_setter',
 'n/086.Irish_setter',
 'n/087.Irish_terrier',
 'n/088.Irish_water_spaniel',
 'n/089.Irish_wolfhound',
 'n/090.Italian_greyhound',
 'n/091.Japanese_chin',
 'n/092.Keeshond',
 'n/093.Kerry_blue_terrier',
 'n/094.Komondor',
 'n/095.Kuvasz',
 'n/096.Labrador_retriever',
 'n/097.Lakeland_terrier',
 'n/098.Leonberger',
 'n/099.Lhasa_apso',
 'n/100.Lowchen',
 'n/101.Maltese',
 'n/102.Manchester_terrier',
 'n/103.Mastiff',
 'n/104.Miniature_schnauzer',
 'n/105.Neapolitan_mastiff',
 'n/106.Newfoundland',
 'n/107.Norfolk_terrier',
 'n/108.Norwegian_buhund',
 'n/109.Norwegian_elkhound',
 'n/110.Norwegian_lundehund',
 'n/111.Norwich_terrier',
 'n/112.Nova_scotia_duck_tolling_retriever',
 'n/113.Old_english_sheepdog',
 'n/114.Otterhound',
 'n/115.Papillon',
 'n/116.Parson_russell_terrier',
 'n/117.Pekingese',
 'n/118.Pembroke_welsh_corgi',
 'n/119.Petit_basset_griffon_vendeen',
 'n/120.Pharaoh_hound',
 'n/121.Plott',
 'n/122.Pointer',
 'n/123.Pomeranian',
 'n/124.Poodle',
 'n/125.Portuguese_water_dog',
 'n/126.Saint_bernard',
 'n/127.Silky_terrier',
 'n/128.Smooth_fox_terrier',
 'n/129.Tibetan_mastiff',
 'n/130.Welsh_springer_spaniel',
 'n/131.Wirehaired_pointing_griffon',
 'n/132.Xoloitzcuintli',
 'n/133.Yorkshire_terrier']

def extract_Xception(tensor):
    from keras.applications.xception import Xception, preprocess_input
    return Xception(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    # img =  tf.keras.utils.load_img(img_path, target_size=(224, 224))
    imgarr = cv2.resize(img_path, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
#     imgten = from_numpy(imgarr)
#     img = imgten.permute((2, 0, 1))
#     img = imgnew.unsqueeze(0)
#     print(type(img), img,shape)
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x =  tf.keras.utils.img_to_array(imgarr)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in img_paths]
    return np.vstack(list_of_tensors)

### a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.
def Xception_predict_breed (img_path):
    # extract the bottle neck features
    bottleneck_feature = extract_Xception(path_to_tensor(img_path))
    ## get a vector of predicted values
    predicted_vector = Xception_model.predict(bottleneck_feature)
    ## return the breed
    return dog_names[np.argmax(predicted_vector)]

def display_img(img_path):
    img = cv2.imread(img_path)
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgplot = plt.imshow(cv_rgb)
    return imgplot

def breed_identifier(img_path):
    response = Xception_predict_breed(img_path)
    prediction = response[6:].replace('_', ' ').title()

    if prediction:
        print (prediction)
        return prediction
    else:
        return print('The picture does not contain a dog')

# YOLO Stuff

def dog_crop(model, tensor):
    # x = image.imread(x)
    results = model(tensor)
    cropped_images = []
    for i in results[0].boxes.data:
        tag = int(i[-1])
        if tag == 16 or tag == 21 or tag == 18:
            conf = float(i[-2])
            if conf > 0.5:
                x1, x2 = int(i[0]),  int(i[2])
                y1, y2 = int(i[1]),  int(i[3])
                cropped_images.append(tensor[y1:y2, x1:x2,:])
#     list_of_imgs = []
#     for i in range(len(cropped_images)):
#         cv2.imwrite(f'imgs/img{i}.jpg', cv2.cvtColor(cropped_images[i], cv2.COLOR_RGB2BGR))
#         list_of_imgs.append(f'imgs/img{i}.jpg')
    return cropped_images

def return_breeds(model, tensor):
    responses = dog_crop(model, tensor)
    rgb_values, dogbreed = [], []
    for i in responses:
        # display_img(i)
        breedtype = breed_identifier(i)
        rgb_values.append(i) # plt.imread(i))
        dogbreed.append(breedtype)
        plt.show()
    return [rgb_values, dogbreed]

from tensorflow import keras
from ultralytics import YOLO

def load_Xception_model():
    Xception_model = keras.models.load_model('Xception_model.h5')
    return Xception_model

def load_YOLO_model():
    model = YOLO("yolov8m.pt")
    return model

model = load_YOLO_model()
Xception_model = load_Xception_model()
