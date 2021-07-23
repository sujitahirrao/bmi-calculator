from src.predict_bmi_from_face_image.model import get_model
from src.predict_bmi_from_face_image.train import train_top_layer

if __name__ == '__main__':
    model = get_model(ignore_age_weights=True)
    model.load_weights("models/bmi_model_weights.h5")
    print('Loaded weights from bmi regressor')
    train_top_layer(model)
    # train_all_layers(model)
