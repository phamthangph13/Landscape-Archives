import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 1. Load và chuẩn bị dữ liệu

def load_images(folder, target_size=(64, 64)):
    if not os.path.exists(folder):
        raise FileNotFoundError(f"The folder '{folder}' does not exist.")
    
    images = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            img_path = os.path.join(folder, filename)
            img = load_img(img_path, target_size=target_size)
            img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
            images.append(img_array)
    
    if not images:
        raise ValueError(f"No images found in the folder '{folder}'.")
    
    return np.array(images)

def mask_images(images, mask_size=(16, 16)):
    masked_images = images.copy()
    masks = []
    for img in masked_images:
        h, w, _ = img.shape
        top = np.random.randint(0, h - mask_size[0])
        left = np.random.randint(0, w - mask_size[1])
        mask = np.ones_like(img)
        mask[top:top+mask_size[0], left:left+mask_size[1], :] = 0
        img[top:top+mask_size[0], left:left+mask_size[1], :] = 0
        masks.append(mask)
    return masked_images, np.array(masks)

# 2. Xây dựng mô hình Generator và Discriminator

def build_generator(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, kernel_size=4, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", activation="relu")(x)
    outputs = layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding="same", activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs)

def build_discriminator(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, kernel_size=4, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs)

# 3. Huấn luyện GAN

def train_gan(generator, discriminator, gan, dataset, epochs=100, batch_size=64):
    for epoch in range(epochs):
        for i in range(0, len(dataset), batch_size):
            real_images = dataset[i:i+batch_size]

            # Mask và tạo ảnh giả
            masked_images, masks = mask_images(real_images)
            fake_images = generator.predict(masked_images)

            # Huấn luyện Discriminator
            discriminator.trainable = True
            discriminator.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            real_labels = np.ones((len(real_images), 1))
            fake_labels = np.zeros((len(real_images), 1))

            d_loss_real = discriminator.train_on_batch(real_images, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)

            # Đóng băng Discriminator khi huấn luyện Generator
            discriminator.trainable = False
            g_loss = gan.train_on_batch(masked_images, real_labels)

        print(f"Epoch {epoch+1}/{epochs}, D Loss Real: {d_loss_real[0]}, "
              f"D Loss Fake: {d_loss_fake[0]}, G Loss: {g_loss}")

# 4. Main program

def main():
    # Load dữ liệu
    folder = "archive"  # Thay bằng đường dẫn đến thư mục ảnh của bạn
    images = load_images(folder)
    print(f"Loaded {len(images)} images")

    # Tạo mô hình
    input_shape = (64, 64, 3)
    generator = build_generator(input_shape)
    discriminator = build_discriminator(input_shape)

    # Biên dịch Discriminator
    discriminator.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Đóng băng Discriminator khi huấn luyện Generator
    discriminator.trainable = False

    gan_input = tf.keras.Input(shape=input_shape)
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)
    gan = tf.keras.Model(gan_input, gan_output)

    gan.compile(optimizer="adam", loss="binary_crossentropy")

    # Huấn luyện GAN
    train_gan(generator, discriminator, gan, images, epochs=1, batch_size=16)

    # Lưu mô hình
    generator.save("generator_model.h5")
    discriminator.save("discriminator_model.h5")


if __name__ == "__main__":
    main()