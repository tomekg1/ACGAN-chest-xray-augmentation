import streamlit as st
import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import io
import os
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

# Fixed class order
fixed_class_order = [
    #'Atelectasis',
    #'Cardiomegaly',
    #'Consolidation',
    #'Edema',
    
    'Effusion',
    #'Emphysema',
    #'Fibrosis',
    #'Hernia',
    #'Infiltration',
    #'Mass',
    'No Finding'
    #'Nodule',
    #'Pleural_Thickening',
    #'Pneumonia',
    #'Pneumothorax'
]

# Parameters
latent_dim = 100
num_classes = len(fixed_class_order)
img_shape = (3, 128, 128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ACGAN Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        self.init_size = img_shape[1] // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_shape[0], 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_embedding(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

@st.cache_resource
def load_pytorch_model(model_path):
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    return generator

@st.cache_resource
def load_keras_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def generate_images(generator, class_idx, num_images, noise=None):
    if noise is None:
        noise = torch.randn(num_images, latent_dim, device=device)
    labels = torch.full((num_images,), class_idx, dtype=torch.long, device=device)
    with torch.no_grad():
        generated = generator(noise, labels)
    return generated

def classify_image(model, img):
    # Convert PyTorch tensor to numpy array
    img_array = np.expand_dims(img, axis=0)
    prediction = model.predict(img_array, verbose=0)

    if len(fixed_class_order) == 2:
        prob = prediction[0][0]
        predicted_class = int(prob >= 0.5)
        confidence = prob * 100 if predicted_class == 1 else (1 - prob) * 100
        return predicted_class, confidence, [1 - prob, prob]
    
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    return predicted_class, confidence, prediction[0]

def tensor_to_image(tensor):
    tensor = tensor.cpu().permute(1, 2, 0).numpy()
    tensor = (tensor * 127.5 + 127.5).astype(np.uint8)
    return Image.fromarray(tensor)

def preprocess_uploaded_image(uploaded_file):
    img = Image.open(uploaded_file)
    img = img.resize((128, 128))
    #img_array = np.array(img)
    
    # Convert to tensor and normalize
    #if img_array.ndim == 2:  # Grayscale
    #    img_array = np.stack((img_array,) * 3, axis=-1)
    return img

# Main app
st.title('ACGAN Image Generator with CNN Classifier')

# Sidebar for model uploads
with st.sidebar:
    st.header("Model Configuration")
    generator_file = st.file_uploader("Upload generator.pth", type="pth")
    classifier_file = st.file_uploader("Upload classifier.keras", type="keras")

# Initialize session state
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []

# Load models
generator = None
classifier = None

if generator_file:
    with open("temp_generator.pth", "wb") as f:
        f.write(generator_file.getbuffer())
    generator = load_pytorch_model("temp_generator.pth")
    os.remove("temp_generator.pth")

if classifier_file:
    with open("temp_classifier.keras", "wb") as f:
        f.write(classifier_file.getbuffer())
    classifier = load_keras_model("temp_classifier.keras")
    os.remove("temp_classifier.keras")

# Tab layout
tab1, tab2 = st.tabs(["Generate Images", "Upload Images"])

with tab1:
    st.header("Image Generation")
    
    if generator is None:
        st.warning("Please upload a generator model first")
    else:
        col1, col2 = st.columns(2)
        with col1:
            selected_class = st.selectbox("Select class to generate:", fixed_class_order)
            class_idx = fixed_class_order.index(selected_class)
        
        with col2:
            num_images = st.slider("Number of images:", 1, 10, 1)
            seed = st.number_input("Seed (optional):", min_value=0, value=None)
        
        if st.button("Generate Images"):
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
            
            noise = torch.randn(num_images, latent_dim, device=device)
            generated_images = generate_images(generator, class_idx, num_images, noise)
            st.session_state.generated_images = generated_images
            
            st.success(f"Generated {num_images} images of class: {selected_class}")
    
    # Display generated images with classification buttons
    if len(st.session_state.generated_images) > 0:
        st.header("Generated Images")
        for i, img_tensor in enumerate(st.session_state.generated_images):
            img = tensor_to_image(img_tensor)
            
            col1, col2 = st.columns([0.5, 0.5])
            with col1:
                st.image(img, caption=f"Generated Image {i+1} - Class: {fixed_class_order[class_idx]}", 
                         use_container_width=True)
            
            with col2:
                # Download button
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                st.download_button(
                    label="Download",
                    data=buf.getvalue(),
                    file_name=f"generated_{fixed_class_order[class_idx]}_{i+1}.png",
                    mime="image/png",
                    key=f"download_{i}"
                )
                
                # Classify button
                if classifier:
                    if st.button(f"Classify Image {i+1}", key=f"classify_{i}"):
                        with st.spinner('Classifying...'):
                            pred_class, confidence, probs = classify_image(classifier, img)
                            
                            st.write(f"**True Class:** {fixed_class_order[class_idx]}")
                            st.write(f"**Predicted Class:** {fixed_class_order[pred_class]}")
                            st.write(f"**Confidence:** {confidence:.2f}%")
                            
                            # Probability distribution
                            fig, ax = plt.subplots(figsize=(8, 3))
                            print(probs)
                            ax.barh(fixed_class_order, np.array(probs)*100)
                            ax.set_xlim(0, 100)
                            ax.set_xlabel('Probability (%)')
                            ax.set_title('Class Probabilities')
                            st.pyplot(fig)

with tab2:
    st.header("Upload Images for Classification")
    
    if classifier is None:
        st.warning("Please upload a classifier model first")
    else:
        uploaded_files = st.file_uploader("Choose images to classify", 
                                         type=["jpg", "jpeg", "png"], 
                                         accept_multiple_files=True)
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    img = preprocess_uploaded_image(uploaded_file)
                    #img = tensor_to_image(img_tensor)
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.image(img, caption=f"Uploaded: {uploaded_file.name}", 
                                 use_container_width=True)
                    
                    with col2:
                        if st.button(f"Classify {uploaded_file.name}", key=f"uploaded_{uploaded_file.name}"):
                            with st.spinner('Classifying...'):
                                pred_class, confidence, probs = classify_image(classifier, img)
                                
                                st.write(f"**Predicted Class:** {fixed_class_order[pred_class]}")
                                st.write(f"**Confidence:** {confidence:.2f}%")
                                # Probability distribution
                                fig, ax = plt.subplots(figsize=(16, 8))
                                ax.barh(fixed_class_order, np.array(probs)*100)
                                ax.set_xlim(0, 100)
                                ax.set_xlabel('Probability (%)')
                                ax.set_title('Class Probabilities')
                                st.pyplot(fig)
                
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")