
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(train_dir, test_dir, img_size=(224, 224)):
    datagen = ImageDataGenerator(rescale=1./255)
    
    train_data = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        class_mode='categorical',
        batch_size=32
    )

    test_data = datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        class_mode='categorical',
        batch_size=32
    )

    return train_data, test_data
