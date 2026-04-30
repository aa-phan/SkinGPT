import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from product_db.match_engine import SkincareMatchingEngine

# 1. ARCHITECTURE DEFINITION
def get_resnet18_model():
    model = models.resnet18(weights=None) # We will load our own weights
    num_ftrs = model.fc.in_features
    # Your project notes specify a 5-class output with Sigmoid for probabilities
    model.fc = nn.Linear(num_ftrs, 5)
    return model

# 2. CACHING THE MODEL
# Using st.cache_resource ensures the .pth file is only loaded into memory once
@st.cache_resource
def load_model():
    model = get_resnet18_model()
    # Ensure five_class_best.pth is in your project root directory
    model.load_state_dict(torch.load("five_class_best.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# 3. PREPROCESSING PIPELINE
def preprocess_image(image):
    # Standard ResNet preprocessing: Resize, Crop, Tensor, Normalize
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0) # Add batch dimension

# --- STREAMLIT UI ---
st.title("Skincare Analysis: Model Integration")
st.write("Upload a photo to generate your Condition Vector.")

# Load model once
model = load_model()
classes = ['Acne', 'Eyebags', 'Hyperpigmentation', 'Wrinkles', 'Dryness']

uploaded_file = st.file_uploader("Upload Skin Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Target Image", width=300)
    
    if st.button("Run Analysis"):
        with st.spinner("Analyzing..."):
            # A. Preprocess
            input_tensor = preprocess_image(img)
            
            # B. Feed the Model
            with torch.no_grad():
                output_vector = model(input_tensor)[0].tolist()
            
            # C. Output Vector Results
            st.subheader("Generated Condition Vector")
            st.json(dict(zip(classes, output_vector)))
            
            # Visualizing the vector for the team
            st.bar_chart(dict(zip(classes, output_vector)))
            
            # D. PRODUCT MATCHING ALGORITHM
            st.subheader("Personalized Skincare Recommendations")
            
            # Append 0 for eyebags (not detected by model)
            user_vector = output_vector + [0]
            
            # Check if product database exists
            db_path = Path("dataset/final_sephora_database.csv")
            if not db_path.exists():
                st.error(f"❌ Product database not found at {db_path}. Please ensure the CSV is present or upload it.")
            else:
                try:
                    # Initialize the matching engine
                    engine = SkincareMatchingEngine(str(db_path))
                    
                    # Get price tier from user
                    price_tier = st.selectbox("Select Your Budget", ["Budget", "Mid-Range", "Premium", "Luxury"])
                    
                    # Run the matching algorithm
                    routine = engine.build_routine(user_vector, price_tier)
                    
                    # Display Results
                    if isinstance(routine, str):
                        # Error message returned by the engine
                        st.warning(routine)
                    else:
                        st.success("✅ Routine generated successfully!")
                        for step, product in routine.items():
                            with st.expander(f"🟩 {step}"):
                                if isinstance(product, dict):
                                    st.metric("Product Name", product.get('product_name', 'Unknown'))
                                    col1, col2, col3 = st.columns(3)
                                    col1.metric("Brand", product.get('brand_name', 'Unknown'))
                                    col2.metric("Price", f"${product.get('price_usd', 0.0):.2f}")
                                    col3.metric("Rating", f"{product.get('rating', 0.0):.1f}/5.0")
                                    st.metric("Match Score", f"{product.get('match_percent', 0.0):.1f}%")
                                    st.write(f"**Active Ingredients:** {', '.join(product.get('clean_ingredient_array', [])[:5])}")
                                else:
                                    st.write(product)
                except Exception as e:
                    st.error(f"❌ Error running matching algorithm: {e}")