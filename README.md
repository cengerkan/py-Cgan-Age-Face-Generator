# Age-Conditional Face Generator using Conditional GAN (CGAN)

A deep learning model built with Conditional GAN (CGAN) architecture to generate synthetic face images based on specified age group labels. This repository showcases how label-conditioned noise vectors can be used to control output generation in a generative adversarial network.

---

## 🎯 Projenin Amacı

Bu proje, yaş grubu etiketlerine göre koşullandırılmış sentetik yüz görüntüleri üretmek amacıyla geliştirilmiştir. Hedef, veri artırımı, yapay veri üretimi ve yaşa duyarlı görüntü sentezi gibi çeşitli bilgisayarla görü uygulamalarında kullanılabilecek bir model sunmaktır. Eğitimli model, farklı yaş gruplarını temsil eden etiketleri girdi olarak alır ve bu etiketlere uygun yüz görüntüleri üretir.

---

## 🧠 Kullanılan Teknikler

- **Conditional GAN (CGAN):** Etiketli veri ile koşullandırılmış görüntü üretimi.
- **Label Embedding:** Kategorik etiketlerin yoğun vektör temsilleriyle birleştirilmesi.
- **Noise Vector Concatenation:** Rastgele gürültü ile etiket bilgisinin birleştirilmesi.
- **LeakyReLU & Batch Normalization:** Daha kararlı ve hızlı öğrenme süreci için.
- **Tanh Output Activation:** -1 ile 1 arası normalize edilmiş çıkışlar.
- **Keras (TensorFlow Backend):** Yüksek seviyeli model geliştirme.

---

## 💾 Eğitim Detayları ve Hiperparametreler

| Parametre             | Değer              |
|-----------------------|--------------------|
| Görüntü Boyutu        | 64×64 RGB          |
| Latent Vektör (z)     | 100 boyut          |
| Etiket Sayısı         | 5 (Yaş grubu)      |
| Optimizer             | Adam (lr=0.0002, β₁=0.5) |
| Batch Size            | 32                 |
| Epoch Sayısı          | 100000             |
| Aktivasyonlar         | LeakyReLU, Tanh    |

> Eğitim sırasında üretici ve ayrımcı modeller birbirine karşı öğrenerek, her adımda daha gerçekçi yüz görüntüleri üretmeyi hedefler.

---

## 🧪 Üretilen Örnek Görseller

Model, aşağıdaki gibi etiketlenmiş yaş gruplarına karşılık gelen yüz görüntülerini üretmektedir:

| Etiket | Açıklama         | Örnek Üretim |
|--------|------------------|--------------|
| 0      | Çocuk (0-12)     | 🧒 ![child](example_images/age_0.png) |
| 1      | Genç (13-25)     | 👩 ![young](example_images/age_1.png) |
| 2      | Yetişkin (26-40) | 👨 ![adult](example_images/age_2.png) |
| 3      | Orta Yaş (41-60) | 👩‍🦳 ![middle](example_images/age_3.png) |
| 4      | Yaşlı (60+)      | 👴 ![elder](example_images/age_4.png) |

> Not: Görseller eğitim sürecinden periyodik olarak kaydedilen örnekleri temsil etmektedir.

---

## 🚀 Kurulum ve Kullanım Talimatları

### Gereksinimler

```bash
pip install numpy matplotlib tensorflow


### MODELİ EĞİTMEK İÇİN 
from cgan_age import train

train(epochs=100000, batch_size=32, sample_interval=200)


### Örnek Görüntü Üretimi
from cgan_age import build_generator
import numpy as np
import matplotlib.pyplot as plt

generator = build_generator()
noise = np.random.normal(0, 1, (1, 100))
label = np.array([[2]])  # 26-40 yaş aralığı
generated_image = generator.predict([noise, label])

plt.imshow((generated_image[0] + 1) / 2)
plt.axis('off')
plt.show()


### Proje Yapısı
.
├── cgan_age.py              # Tüm model ve eğitim fonksiyonları
├── example_images/          # Üretilmiş örnek görseller
├── README.md                # Proje dökümantasyonu




