from model import get_model
from train import train_top_layer

if __name__ == '__main__':
    model = get_model()
    train_top_layer(model)
    # train_all_layers(model)
