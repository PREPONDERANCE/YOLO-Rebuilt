import torch
import torchvision.transforms as transforms

from PIL import Image
from models import YOLOv1
from loss import SSELoss


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
image_tensor = image_tensor.unsqueeze(0)

model = YOLOv1()
loss_fn = SSELoss()
optimzer = torch.optim.Adam(model.parameters(), lr=1e-4)

pred = model(image_tensor)
loss = loss_fn(pred, torch.zeros_like(pred))

print(loss.item())
