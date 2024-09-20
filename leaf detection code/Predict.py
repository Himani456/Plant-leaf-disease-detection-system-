  import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def predict_image(model_path, img_path):
    model = load_model(model_path)

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    return np.argmax(predictions)

if __name__ == '__main__':
    result = predict_image('models/model.h5', 'path_to_your_image.jpg')
    print(f'Predicted class: {result}')
  
  
