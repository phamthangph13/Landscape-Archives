import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

# Tải mô hình Generator đã huấn luyện
def load_generator(model_path):
    return tf.keras.models.load_model(model_path)

# Tải và xử lý ảnh đầu vào
def load_image(image_path, target_size=(64, 64)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Chuẩn hóa về [0, 1]
    return img_array

# Tạo mask cho ảnh đầu vào
def create_masked_image(image, mask_size=(16, 16)):
    masked_image = image.copy()
    h, w, _ = masked_image.shape
    top = np.random.randint(0, h - mask_size[0])
    left = np.random.randint(0, w - mask_size[1])
    mask = np.ones_like(masked_image)
    mask[top:top+mask_size[0], left:left+mask_size[1], :] = 0
    masked_image[top:top+mask_size[0], left:left+mask_size[1], :] = 0
    return masked_image, mask

# Tái tạo ảnh từ mô hình Generator
def reconstruct_image(generator, masked_image):
    input_image = np.expand_dims(masked_image, axis=0)  # Thêm batch dimension
    reconstructed_image = generator.predict(input_image)
    return reconstructed_image[0]  # Bỏ batch dimension

# Lưu ảnh ra file
def save_image(image, output_path):
    img = array_to_img(image * 255.0)  # Chuyển về giá trị [0, 255]
    img.save(output_path)

# Chương trình chính
def main():
    # Đường dẫn ảnh và mô hình
    input_image_path = "test.jpg"  # Thay bằng đường dẫn ảnh đầu vào
    output_image_path = "reconstructed_image.jpg"
    model_path = "generator_model.h5"

    # Tải mô hình Generator
    generator = load_generator(model_path)
    print("Generator model loaded.")

    # Tải và xử lý ảnh đầu vào
    original_image = load_image(input_image_path)
    masked_image, mask = create_masked_image(original_image)
    print("Masked image created.")

    # Tái tạo ảnh
    reconstructed_image = reconstruct_image(generator, masked_image)
    print("Image reconstructed.")

    # Lưu ảnh kết quả
    save_image(reconstructed_image, output_image_path)
    print(f"Reconstructed image saved to {output_image_path}.")

if __name__ == "__main__":
    main()
