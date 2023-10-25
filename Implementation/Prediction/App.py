import streamlit as st
import pandas as pd
import os
import pydicom
import tkinter as tk
from tkinter import filedialog
from model import model_run
import numpy as np


# Function to open a folder dialog
def browse_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()
    return folder_path

# Add a header section in dark blue
st.markdown('<h1 style="text-align: left; padding: 0px 0px 10px 0px; ;">Get an analysis for your patient</h1>', unsafe_allow_html=True)


st.markdown('''
This is a supporting tool designed specifically for medical professionals. Built upon the cutting-edge prototypical network architecture, this tool boasts an accuracy rate of 92%. It offers two valuable predictions in alignment with the training it has undergone. It's essential to emphasize that our AI model is a complementary assisting tool, enhancing the capabilities of medical experts and facilitating their decision-making processes, ultimately improving patient care.
            ''')


folder_path = 'C:/Users/Nimesha/Documents/MSC_RESEARCH/Implementation/Prediction/data'

for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)


st.markdown(
    """
    <style>
    .text1 {
        font-size: 18px; 
    }

     .text2 {
        font-size: 18px; 
       
    }

     .text3 {
        font-size: 19px; 
        color: green;
    }

     .text4 {
        font-size: 18px; 
        
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Create a file uploader widget and apply the class name
st.write("<span class='text1'><b>Upload the patient's CT scan file</b></span>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["dcm"])


if uploaded_file is not None:
    # Process the uploaded file
    

    file_path = os.path.join("data", uploaded_file.name)


    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.write("<span class='text2'>File was successfully uploaded</span>", unsafe_allow_html=True)

    

    results = model_run()
    print(results[0][1])

    if results[0][0] == 0 :
        st.write("<span class='text3'>Predition : <b>class A</b></span>", unsafe_allow_html=True)
        st.write("<span class='text4'>It is highly probale that the patient has Adenocarcinoma accoring the model</span>", unsafe_allow_html=True)
       
    elif results[0][0] == 1 :
         st.write("<span class='text3'>Predition : class B</span>", unsafe_allow_html=True)
         st.write("<span class='text4'>It is highly probale that the patient has Small Cell Carcinoma according the model </span>", unsafe_allow_html=True)
    
    elif results[0][0] == 2 :
         st.write("<span class='text3'>Predition : class G</span>", unsafe_allow_html=True)
         st.write("<span class='text4'>It is highly probale that the patient has Large Cell Carcinoma according the model</span>", unsafe_allow_html=True)

    elif results[0][0] == 3 :
        st.write("<span class='text3'>Predition : class E</span>", unsafe_allow_html=True)
        st.write("<span class='text4'>It is highly probale that the patient has Squamous Cell Carcinoma according the model</span>", unsafe_allow_html=True)

    
  
    if results[0][1] == 0 :
        st.write("<span class='text4'>There is a slight likelihood that the patient might have Adenocarcinoma according to the model.</span>", unsafe_allow_html=True)
       
    elif results[0][1] == 1 :
         st.write("<span class='text4'>There is a slight likelihood that the patient might have Small Cell Carcinoma according to the model </span>", unsafe_allow_html=True)
    
    elif results[0][1] == 2 :
         st.write("<span class='text4'>There is a slight likelihood that the patient might have Large Cell Carcinoma according to the model</span>", unsafe_allow_html=True)

    elif results[0][1] == 3 :
         st.write("<span class='text4'>There is a slight likelihood that the patient might have Squamous Cell Carcinoma according to the model</span>", unsafe_allow_html=True)


    # Read and display the DICOM file
    ds = pydicom.dcmread(file_path)
    if hasattr(ds, "pixel_array"):
        # Adjust pixel values to be within the [0, 255] range
        pixel_array = ds.pixel_array
        pixel_array = pixel_array.astype(np.float32)
        pixel_array = pixel_array - np.min(pixel_array)
        pixel_array = pixel_array / np.max(pixel_array) * 255.0
        pixel_array = pixel_array.astype(np.uint8)
        
        st.image(pixel_array, width=400, channels="GRAY")
    else:
        st.write("This DICOM file does not contain image data.")