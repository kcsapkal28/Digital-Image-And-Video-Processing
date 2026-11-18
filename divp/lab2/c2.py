
import cv2
import matplotlib.pyplot as plt
import os

print("========== PROGRAM STARTED ==========")

# Use the complete path to your image
image_path = r"C:\Users\admin\OneDrive\Desktop\divp\lab2\lion.wall.png"

print("Image path:", image_path)
print("File exists:", os.path.exists(image_path))

# Load image
image = cv2.imread(image_path)

if image is None:
    print("ERROR: Image could not be loaded.")

else:
    print("Image loaded successfully!")
    print("Image shape:", image.shape)

    # Convert BGR to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Separate channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_clahe = clahe.apply(l)

    # Merge channels
    lab_clahe = cv2.merge((l_clahe, a, b))

    # Convert LAB to BGR
    enhanced = cv2.cvtColor(
        lab_clahe,
        cv2.COLOR_LAB2BGR
    )

    # Save enhanced image separately
    output_path = r"C:\Users\admin\OneDrive\Desktop\divp\lab2\lion_enhanced.png"

    success = cv2.imwrite(output_path, enhanced)

    print("Image saved:", success)
    print("Output path:", output_path)

    # Display original and enhanced images
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    plt.title("CLAHE Enhanced Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

print("========== PROGRAM FINISHED ==========")

