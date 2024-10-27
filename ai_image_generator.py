# Source: @DeepCharts Youtube Channel (https://www.youtube.com/@DeepCharts)

# PART 1: LIBRARY IMPORTS

import streamlit as st
import replicate
import os
import requests
from PIL import Image
from io import BytesIO


# PART 2: SETUP REPLICATE CREDENTIALS AND AUTHENTICATION

# Set up your Replicate API key (optionally from environment variable)
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")  # You can store your API key in an environment variable

if REPLICATE_API_TOKEN is None:
    st.error("Replicate API token not found. Please set it in your environment.")
    st.stop()

# Authenticate with Replicate using the API token
replicate.Client(api_token=REPLICATE_API_TOKEN)


# PART 3: STREAMLIT WEBAPP

# Initialize session state for storing the generated image URL
if 'image_url' not in st.session_state:
    st.session_state['image_url'] = None


# PART 3A: SIDEBAR OPTIONS

# Sidebar inputs
with st.sidebar:

    # Title of the app
    st.title('AI Image Generation: Flux Schnell')

    st.header("Prompt and Options")

    # Input box for the user to type the prompt (using text_area for multiline input)
    prompt = st.text_area('Enter a prompt to generate an image', height=50)

    # Checkbox to enable or disable random seed
    use_random_seed = st.checkbox('Use Random Seed', value=True)

    # Slider for random seed (only if the checkbox is checked)
    if use_random_seed:
        random_seed = st.slider('Random Seed', 0, 1000, 435)
    else:
        random_seed = None

    # Slider for output quality
    output_quality = st.slider('Output Quality', 50, 100, 80)

    # Create two columns for Generate and Download buttons
    col1, col2 = st.columns([1, 1])

    # Button to submit the prompt and generate image
    generate_button = col1.button('Generate Image')


# PART 4A: MAIN CONTENT AREA (IMAGE GENERATION AND ACCESS)

# Check if the button was pressed and if there is a prompt
if generate_button and prompt:
    with st.spinner('Generating image...'):
        try:
            # Call the Flux Schnell model on Replicate
            input_data = {
                "prompt": prompt,
                "aspect_ratio": '3:2',  # Set the aspect ratio
                "quality": output_quality  # Set the output quality
            }

            # Add random seed only if it's enabled
            if random_seed is not None:
                input_data["seed"] = random_seed

            # Use replicate.run to invoke the model
            output = replicate.run(
                "black-forest-labs/flux-schnell",  # Model name
                input=input_data  # Input to the model
            )

            # Store the generated image URL in session state
            st.session_state['image_url'] = output[0]  # Assuming the image is the first element in output

        except Exception as e:
            st.error(f"An error occurred: {e}")

# If an image URL is present in session state, display the image and download button
if st.session_state['image_url']:
    # Display the image
    st.image(st.session_state['image_url'], caption='Generated Image')

    # Download the image from the URL
    response = requests.get(st.session_state['image_url'])
    image = Image.open(BytesIO(response.content))

    # Convert the image to a binary stream and save it as .jpg
    img_buffer = BytesIO()
    image.save(img_buffer, format="JPEG") 
    img_buffer.seek(0)

    # Display the download button in the second column
    with col2:
        st.download_button(
            label="Download Image",
            data=img_buffer,
            file_name="generated_image.jpg",
            mime="image/jpeg"
        )