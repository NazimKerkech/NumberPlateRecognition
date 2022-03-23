import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2
import chemins

def app():
    
    # Sidebar
    st.sidebar.write('# Select an image')
    uploaded_file = st.sidebar.file_uploader('',
                                             type=['png', 'jpg', 'jpeg'],
                                             accept_multiple_files=False)
    
    st.sidebar.write("# Localisation")
    confidence_threshold_loc = st.sidebar.slider('Seuil de la confidence pour la localisation :', 0.0, 1.0, 0.5, 0.01)
    iou_threshold_loc = st.sidebar.slider('Seuil de l\'IoU pour la localisation :', 0.0, 1.0, 0.5, 0.01)
    
    st.sidebar.write("# Reconnaissance")
    confidence_threshold_reco = st.sidebar.slider('Seuil de la confidence pour la reconnaissance :', 0.0, 1.0, 0.5, 0.01)
    iou_threshold_reco = st.sidebar.slider('Seuil de l\'IoU pour la reconnaissance :', 0.0, 1.0, 0.5, 0.01)
    
    # Appercu
    if uploaded_file is None:
        # Default image.
        image = cv2.imread(chemins.dataset_localisation+"car_3049.jpg")
    
    else:
        # User-selected image.
        image = np.array(Image.open(uploaded_file))
    
    localisation = st.container()
    reconnaissance = st.container()
    chiffres = st.container()
    
    with localisation:
        st.markdown("# Localisation")
    
        # Detection avec YOLOv5
        model_loc = torch.hub.load(chemins.chemin_vers_yolo, "custom",
                                path="localisation.pt",
                                source="local")
        model_loc.conf = confidence_threshold_loc    # NMS (Non Maximum Suppression) confidence threshold
        model_loc.iou = iou_threshold_loc            # NMS IoU threshold
    
        results = model_loc(image)
    
        res_loc = results.xyxy[0]
    
        image_pour_loc = image.copy()
        for i in range(len(res_loc)):
            cv2.rectangle(image_pour_loc, (int(res_loc[i][0]), int(res_loc[i][1])), (int(res_loc[i][2]), int(res_loc[i][3])), (255, 0, 0))
    
        # Display image.
        st.image(image_pour_loc)
    
        col1, col2 = st.columns(2)
        col1.markdown("Precision")
        col2.write(1)
     
        col1.markdown("Rappel")
        col2.write(1)
    
        col1.markdown("F-score")
        col2.write(1)
    
    with reconnaissance:
        st.markdown("# Reconnaissance")
    
        img_plaque = image[int(res_loc[0][1]) : int(res_loc[0][0]), int(res_loc[0][3]) : int(res_loc[0][2])]
        img_plaque = cv2.resize(img_plaque, (1000, int(img_plaque.shape[0]/img_plaque.shape[1]*1000)))
    
        model_reco = torch.hub.load(chemins.chemin_vers_yolo, "custom",
                                path="reconnaissance.pt",
                                source="local")
    
        model_reco.conf = confidence_threshold_reco    # NMS (Non Maximum Suppression) confidence threshold
        model_reco.iou = iou_threshold_reco            # NMS IoU threshold
    
        results_reconaissance = model_reco(img_plaque)
        res_rec = sorted(results_reconaissance.xyxy[0], key=lambda x: x[1], reverse=True)
    
        image_pour_reco = img_plaque.copy()
    
        images_chiffres = []
        contenu = []
        for i in range(len(res_rec)):
            images_chiffres.append(img_plaque[int(res_rec[i][1]) : int(res_rec[i][3]), int(res_rec[i][0]) : int(res_rec[i][2])])
            contenu.append(int(res_rec[i][5]))
            cv2.rectangle(image_pour_reco, (int(res_rec[i][0]), int(res_rec[i][1])), (int(res_rec[i][2]), int(res_rec[i][3])), (255, 0, 0))
    
        st.image(image_pour_reco, width=1000)
    
        st.markdown("Contenu de la plaque : ")
        st.write(contenu)
    
        col1, col2 = st.columns(2)
        col1.markdown("Precision")
        col2.write(1)
    
        col1.markdown("Rappel")
        col2.write(1)
    
        col1.markdown("F-score")
        col2.write(1)
    
    with chiffres:
        st.markdown("# Chiffres")
    
        for i in range(len(res_rec)):
            col1_c, col2_c = st.columns(2)
            col1_c.image(images_chiffres[i], width=100)
            col2_c.markdown("Chiffre : " + str(int(res_rec[i][5])))
            col2_c.markdown("Confidence : " + str(float(res_rec[i][4])))
