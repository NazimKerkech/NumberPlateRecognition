
# app.py, run with 'streamlit run app.py'
import pandas as pd
import sys
import streamlit as st
import matplotlib.pyplot as plt
from utility import *

#cmd: streamlit run C:\M2_work\pfe\localisation_part\my_app\\app.py

train_csv="train_predictions_17_03_2022.csv"
val_csv="validation_predictions_17_03_2022.csv"

def app():
	########################## LEFT SIDE OF APP ##########################
	page_option = st.sidebar.selectbox(
	    'Choose an option',
	     ('View Data','View Model')
	     )

	########################## RIGHT SIDE OF APP ##########################

	afficher_page(page_option)

	########################## 
