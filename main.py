import utils
import train
import models
from tensorflow.keras.datasets.fashion_mnist import load_data

if __name__ == "__main__":
    # Load data:
    (X_train, y_train), (X_test, y_test) = load_data()

    # Kích thước latent space
    latent_dim = 100
    # Khởi tạo discriminator
    d_model = models._discriminator()
    # Khởi tạo generator
    g_model = models._generator(latent_dim)
    # Khởi tạo cgan
    cgan_model = models._cgan(g_model, d_model)
    # load image data
    dataset = utils._standardize_data(X_train, y_train)
    # train model
    train._train(g_model, d_model, cgan_model, dataset, latent_dim)