import cv2
import numpy as np
import matplotlib.pyplot as plt

def negative(img):
    return 255 - img

def logarithmic(img):
    img = img.astype(np.float32)
    c = 255 / np.log(1 + np.max(img))
    result = c * np.log(1 + img)
    return np.uint8(result)

def power_law(img, gamma):
    img = img.astype(np.float32) / 255
    result = np.power(img, gamma)
    return np.uint8(result * 255)

# Use the FULL path to your image
image_path = r"C:\Users\admin\OneDrive\Desktop\divp\lab1\lab1.jpg"

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(
        f"Image could not be loaded.\nCheck that this file exists:\n{image_path}"
    )

print("Image loaded successfully!")
print("Image shape:", img.shape)

# Transformations
neg = negative(img)
log = logarithmic(img)
gamma = power_law(img, 0.5)

# Display
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(neg, cmap="gray")
plt.title("Negative")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(log, cmap="gray")
plt.title("Logarithmic")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(gamma, cmap="gray")
plt.title("Power Law (γ = 0.5)")
plt.axis("off")

plt.tight_layout()
plt.show()
