from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np
import requests

BUCKET_NAME = 'fruit-tf-model'

class_names = ['Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3', 'Apple Granny Smith',
         'Apple Pink Lady', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3', 'Apple Red Delicious', 'Apple Red Yellow 1', 'Apple Red Yellow 2',
         'Apricot', 'Avocado', 'Avocado ripe', 'Banana', 'Banana Lady Finger', 'Banana Red', 'Beetroot', 'Blueberry', 'Cactus fruit',
         'Cantaloupe 1', 'Cantaloupe 2', 'Carambula', 'Cauliflower', 'Cherry 1', 'Cherry 2', 'Cherry Rainier', 'Cherry Wax Black',
         'Cherry Wax Red', 'Cherry Wax Yellow', 'Chestnut', 'Clementine', 'Cocos', 'Corn', 'Corn Husk', 'Cucumber Ripe', 'Cucumber Ripe 2',
         'Dates', 'Eggplant', 'Fig', 'Ginger Root', 'Granadilla', 'Grape Blue', 'Grape Pink', 'Grape White', 'Grape White 2', 'Grape White 3',
         'Grape White 4', 'Grapefruit Pink', 'Grapefruit White', 'Guava', 'Hazelnut', 'Huckleberry', 'Kaki', 'Kiwi', 'Kohlrabi', 'Kumquats',
         'Lemon', 'Lemon Meyer', 'Limes', 'Lychee', 'Mandarine', 'Mango', 'Mango Red', 'Mangostan', 'Maracuja', 'Melon Piel de Sapo',
         'Mulberry', 'Nectarine', 'Nectarine Flat', 'Nut Forest', 'Nut Pecan', 'Onion Red', 'Onion Red Peeled', 'Onion White', 'Orange',
         'Papaya', 'Passion Fruit', 'Peach', 'Peach 2', 'Peach Flat', 'Pear', 'Pear 2', 'Pear Abate', 'Pear Forelle', 'Pear Kaiser',
         'Pear Monster', 'Pear Red', 'Pear Stone', 'Pear Williams', 'Pepino', 'Pepper Green', 'Pepper Orange', 'Pepper Red', 'Pepper Yellow',
         'Physalis', 'Physalis with Husk', 'Pineapple', 'Pineapple Mini', 'Pitahaya Red', 'Plum', 'Plum 2', 'Plum 3', 'Pomegranate', 'Pomelo Sweetie',
         'Potato Red', 'Potato Red Washed', 'Potato Sweet', 'Potato White', 'Quince', 'Rambutan', 'Raspberry', 'Redcurrant', 'Salak', 'Strawberry',
         'Strawberry Wedge', 'Tamarillo', 'Tangelo', 'Tomato 1', 'Tomato 2', 'Tomato 3', 'Tomato 4', 'Tomato Cherry Red', 'Tomato Heart',
         'Tomato Maroon', 'Tomato Yellow', 'Tomato not Ripened', 'Walnut', 'Watermelon']


# dataset = tf.keras.preprocessing.image_dataset_from_directory('C:/Users/jjdun/Documents/Data_Science_Projects/Fruits/fruits360/fruits-360_dataset/fruits-360/Training')
# class_names = dataset.class_names
# print(class_names)

model = None


def nutrient_info(prediction):
    url = "https://food-nutrition-information.p.rapidapi.com/foods/search"
    headers = {
        'x-rapidapi-host': "XXXXXXXXXXXXXXXXXXXX",
        'x-rapidapi-key': "XXXXXXXXXXXXXXXXXXXXXXXXX"
    }

    querystring = {"query": f"{prediction}"}

    response = requests.request("GET", url, headers=headers, params=querystring)
    data = response.json()

    fruit_data = data['foods'][0]['foodNutrients']

    removeitems = ['nutrientId', 'foodNutrientSourceDescription', 'foodNutrientId', 'foodNutrientSourceId',
                   'foodNutrientSourceCode', 'derivationId', 'derivationCode', 'derivationDescription',
                   'nutrientNumber', 'rank', 'indentLevel', 'dataPoints']
    nutrient_data = []

    for entry in fruit_data:
        for key, value in entry.items():
            if key not in removeitems:
                nutrient_data.append({key: value})
    return nutrient_data


# BLOB = binary large object
def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()  # download the fruit model file from cloud
    bucket = storage_client.get_bucket(bucket_name)  # name of the project on gcp
    blob = bucket.blob(source_blob_name)  # where the bucket is located
    blob.download_to_filename(destination_file_name)  # the location where the blob gets downloaded


def predict(request):
    global model
    if model is None:  # this function only needs to be called once
        download_blob(
            BUCKET_NAME,
            'models/model2.h5',
            '/tmp/model2.h5'
        )
        model = tf.keras.models.load_model('/tmp/model2.h5')

    image = request.files['file']
    image = np.array(Image.open(image).convert('RGB').resize((100, 100)))  # converts image to RGB format and rescales the photo if it is not 100,100
    image = image/100  # normalize it to be between 0 and 1
    img_array = tf.expand_dims(image, 0)  # expand dimensions since the predict function is expecting a batch of images

    predictions = model.predict(img_array)
    # print('Predictions:', predictions)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    fruit_nutrients = nutrient_info(predicted_class)
    # , 'nutrients': fruit_nutrients
    return{'class': predicted_class, 'confidence': float(confidence), 'nutrients': fruit_nutrients}
