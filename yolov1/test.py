from PIL import Image
import torchvision.transforms as transforms


def load_and_preprocess_image(image_path):
    # Read the image
    img = Image.open(image_path)

    # Define the transformation
    transform = transforms.Compose(
        [
            transforms.Resize((448, 448)),  # Resize to 448x448
            transforms.ToTensor(),  # Convert to tensor (C x H x W) format
        ]
    )

    img_tensor = transform(img)

    return img_tensor


# Example usage
image_path = "/Users/mac/Downloads/sample-image.jpeg"
image_tensor = load_and_preprocess_image(image_path)

print(image_tensor.shape)
