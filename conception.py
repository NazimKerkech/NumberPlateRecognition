import streamlit as st
from contextlib import contextmanager, redirect_stdout
from io import StringIO
import subprocess
from streamlit_tensorboard import st_tensorboard
import chemins
import os

@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret
        
        stdout.write = new_write
        yield

def save_uploadedfile(uploadedfile):
     with open(os.path.join("tempDir",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return st.success("Saved File:{} to tempDir".format(uploadedfile.name))
 
def app():
    # Sidebar
    model_expander = st.sidebar.expander('Model')
    im_size_expander = st.sidebar.expander('Taille des images')
    epochs_expander = st.sidebar.expander('Nombre d\'iterations')
    batch_size_expander = st.sidebar.expander('Taille des batches')
    yaml_data_file_expander = st.sidebar.expander('Fichier data yaml')
    crop_expander = st.sidebar.expander('Crop')
    
    with model_expander:
        model = model_expander.selectbox("Model", ('YOLOv5 s', 'YOLOv5 m', 'YOLOv5 l', 'YOLOv5 x'))
        model = chemins.chemin_vers_yolo + "/models/yolov5" + model[-1] + ".yaml"
    
    with epochs_expander:
        epochs = epochs_expander.text_input("Nombre d\'iterations", 1)
    
    with im_size_expander:
        im_size = im_size_expander.selectbox("Taille", (104, 208, 416))
    
    with batch_size_expander:
        batch_size = batch_size_expander.selectbox("Taille des batches", (2, 4, 8, 16, 32))
    
    with yaml_data_file_expander:
        yaml_data_file = yaml_data_file_expander.file_uploader('Selectionner le fichier yaml contenant les informations du Dataset',
                                             type=['yaml'],
                                             accept_multiple_files=False)
    
    with crop_expander:
        crop_checkbox = crop_expander.checkbox('Crop')
        if crop_checkbox:
            crop = True
        
        ###############################################
        
    entrainement = st.container()    
    with entrainement:
        st.markdown("# Conception du model")
        
        commencer_lentrainement = entrainement.button('Commencer l\'Entrainement')
        if commencer_lentrainement:
            
            # Sauvgarder le fichier .yaml dans la racine
            # Pour pouvoir l'utiliser directement pour la commande
            with open(os.path.join("", "data.yaml"),"wb") as f:
                f.write(yaml_data_file.getbuffer())        
        
            # Entrainement avec YOLOv5
            yaml_data_file = yaml_data_file.name
            print(yaml_data_file)
            commande= 'python ' + chemins.chemin_vers_yolo + '/train.py --img ' + str(im_size) + ' --batch ' + str(batch_size) + ' --epochs ' + str(epochs) + ' --data data.yaml' + ' --cfg ' + model
            process = subprocess.Popen(commande, stdout=subprocess.PIPE, shell=True, encoding='utf-8')
            output= st.empty()
            with st_capture(output.code):
                while True:
                    realtime_output = process.stdout.readline()
                    if realtime_output == '' and process.poll() is not None:
                        break
                    if realtime_output:
                        print(realtime_output.strip(), flush=True)
                        
            st_tensorboard(logdir=chemins.chemin_vers_yolo+"runs", port=6006, width=1080)
