  import tensorflow as tf
from data_preprocessing import load_data
from model import create_model

def train_model(train_dir, test_dir, epochs=10):
    train_data, test_data = load_data(train_dir, test_dir)
    
    model = create_model(num_classes=len(train_data.class_indices))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, validation_data=test_data, epochs=epochs)
    model.save('models/model.h5')

if __name__ == '__main__':
    train_model('data/train/', 'data/test/', epochs=10)
