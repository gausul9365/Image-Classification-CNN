import torch 
from PIL import Image
from torchvision import transforms
from model import CNN


classes = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
)


transform = transforms.Compose([
  transforms.Resize((32, 32)),
  transforms.ToTensor(),
  transforms.Normalize(
    (0.4914, 0.4822, 0.4465),
    (0.2470, 0.2435, 0.2616)
  )
])

model = CNN()
model.load_state_dict(torch.load("cnn_model.pth"))

model.eval()

image = Image.open("sample.jpg").convert("RGB")
image = transform(image)
image = image.unsqueeze(0)


with torch.no_grad():

  output = model(image)

  _, predicted = torch.max(output, 1)


print(f"Prediction: ", classes[predicted.item()])







