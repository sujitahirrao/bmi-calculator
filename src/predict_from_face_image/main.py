from src.predict_from_face_image.model import get_model
from src.predict_from_face_image.train import train_top_layer

if __name__ == '__main__':
    model = get_model()
    train_top_layer(model)
    # train_all_layers(model)
