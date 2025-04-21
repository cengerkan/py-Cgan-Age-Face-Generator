import numpy as np
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, multiply
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


# Hyperparametreler
img_shape = (64, 64, 3)  # Yüz resimlerinin boyutu (64x64x3 RGB)
num_classes = 5  # Yaş grubu sayısı
latent_dim = 100  # Gürültü boyutu


# Üretici (Generator) Modeli
# Üretici (Generator) Modeli
def build_generator():
    model = Sequential()

    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,), dtype='int32')

    # Label embedding boyutu latent_dim ile aynı olacak şekilde ayarlanır
    label_embedding = Flatten()(Embedding(num_classes, latent_dim)(label))
    model_input = Concatenate()([noise, label_embedding])  # noise + label_embedding ile birleştiriyoruz (100 + 100)

    # Dense katmanları
    model.add(Dense(256, input_dim=latent_dim + latent_dim))  # input_dim=200
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))

    img = model(model_input)

    return Model([noise, label], img)

# Ayrımcı (Discriminator) Modeli
def build_discriminator():
    model = Sequential()

    # Ayrımcı giriş boyutunu ayarlıyoruz
    model.add(Dense(512, input_dim=(np.prod(img_shape) + np.prod(img_shape))))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(1, activation='sigmoid'))

    # Girişleri yeniden düzenleyelim
    img = Input(shape=img_shape)
    label = Input(shape=(1,), dtype='int32')

    # Label embedding ve düzleştirme işlemi
    label_embedding = Flatten()(Embedding(num_classes, np.prod(img_shape))(label))
    flat_img = Flatten()(img)

    # Model girişini oluşturun
    model_input = Concatenate()([flat_img, label_embedding])

    # Ayrımcı çıktısını alın
    validity = model(model_input)

    return Model([img, label], validity)


# Modeli Derleme ve Eğitim
def train(epochs, batch_size=128, sample_interval=50):
    # Optimizer
    optimizer = Adam(0.0002, 0.5)

    # Üretici ve Ayrımcıları oluştur ve derle
    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    generator = build_generator()

    # GAN modeli (üreticiyi eğitirken ayrımcıyı sabit tutuyoruz)
    z = Input(shape=(latent_dim,))
    label = Input(shape=(1,))
    img = generator([z, label])

    discriminator.trainable = False
    valid = discriminator([img, label])

    combined = Model([z, label], valid)
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    # Eğitim döngüsü burada yer alacak
    for epoch in range(epochs):
        # Gerçek ve sahte görüntülerle ayrımcıyı eğitme

        # Üretici ile rastgele yaş etiketlerine göre görüntüler üretme ve eğitme adımları

        # Periyodik olarak numune üretme ve sonuçları gözlemleme

        if epoch % sample_interval == 0:
            print(f"{epoch} epochs completed.")

    generator.save(f'generator_model_epoch_{epoch}.h5')
    discriminator.save(f'discriminator_model_epoch_{epoch}.h5')

# Eğitim parametrelerini belirleyin
train(epochs=100000, batch_size=32, sample_interval=200)

# Modeli oluştur
generator = build_generator()

# Test için rastgele gürültü ve etiket oluştur
noise = np.random.normal(0, 1, (1, latent_dim))  # 1 adet rastgele gürültü
label = np.array([[0]])  # 1. yaş grubunu temsil eden etiket

# Üretici ile görüntü üret
generated_image = generator.predict([noise, label])

# Üretilen görüntüyü görselleştir
plt.imshow((generated_image[0] + 1) / 2)  # -1 ile 1 arası değerleri 0 ile 1 aralığına getir
plt.axis('off')
plt.show()
