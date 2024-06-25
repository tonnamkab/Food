import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import timm

class_labels = ["Baby Back Ribs : Calorie = 360 cal", "Burger King Double Whopper : Calorie = 894 cal", "Chicken Caesar Salad : Calorie = 392 cal", "Fried Shrimp : Calorie = 75 cal", "Meatloaf : Calorie = 721 cal", "Pizza : Calorie = 272 cal", "Ramen : Calorie = 380 cal"]

# Load your trained model checkpoint
model_path = "mobilenetv3_large_100_checkpoint_fold4.pt"

# Define the MobileNetV3 model from timm with the desired precision
model = timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=7).to(dtype=torch.float32)

# Load the checkpoint
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

# Extract the state dictionary from the checkpoint
state_dict = checkpoint.state_dict() if isinstance(checkpoint, torch.nn.Module) else checkpoint

# Load the state dictionary into the model
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Preprocessing steps (same as before)
preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create a function to make predictions
def predict(image):
    img = Image.open(image).convert('RGB')
    img = preprocess(img).unsqueeze(0).float()  # Convert to float32 explicitly
    with torch.no_grad():
        model.eval()
        prediction = model(img)
        predicted_class = torch.argmax(prediction).item()
        return predicted_class

# Streamlit app
st.title('CALORIE FOOD')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make predictions on the uploaded image
    if st.button('Classify'):
        label = predict(uploaded_file)
        menu = class_labels[label]
        st.write(f"Predicted Menu: {menu}")