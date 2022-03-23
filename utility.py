import pandas as pd
import streamlit as st
#from yolov5_master.utils.metrics import box_iou
#import yolov5_master.utils.metrics
import matplotlib.pyplot as plt
import cv2
import chemins


########################## PLOTS FUNCTIONS #########################
def afficher_plot(plot):
    if plot=='True Positives':
        plot_all_iou(chemins.train_csv, chemins.val_csv)
    if plot=='Precision':
        plot_all_precision(chemins.train_csv, chemins.val_csv)
    if plot=='Recall':
        plot_all_recall(chemins.train_csv, chemins.val_csv)
    if plot=='F1score':
        plot_all_f1score(chemins.train_csv, chemins.val_csv)

def afficher_data():
    df_train = pd.read_csv(chemins.train_csv)  # read a CSV file inside the 'data" folder next to 'app.py'
    #df_train.drop(columns=df_train.columns[0], axis=1, inplace=True)
    df_val= pd.read_csv(chemins.val_csv)  # read a CSV file inside the 'data" folder next to 'app.py'
    #df_val.drop(columns=df_val.columns[0], axis=1, inplace=True)
    st.title("TRAIN PREDICTIONS")  # add a title
    st.write(df_train)  # visualize my dataframe in the Streamlit app
    csv_summary(df_train)
    st.title("VALIDATION PREDICTIONS")
    st.write(df_val)
    csv_summary(df_val)

    st.title('Plotting')
    plot=st.selectbox('',
        ('True Positives','Precision','Recall','F1score')
    )
    afficher_plot(plot)

def afficher_page(page_option):
    if page_option=='View Data':
        afficher_data()
    if page_option=='View Model':
        st.title("MODEL page")

########################## DISPLAY IMAGE ##########################
def convert_to_list(a):
    a = a.replace("[", "")
    a = a.replace("]", "")
    a = a.split(", ")
    if len(a)>4:
        return None
    a = [int(float(x.replace("'", ""))) for x in a]
    return a

def get_gt_pred(name):
    df_train = pd.read_csv(chemins.train_csv)
    df_val=pd.read_csv(chemins.val_csv)
    gt=[]
    pred=[]
    x1=()
    x2=()
    y1=()
    y2=()
    for ind in df_train.index:
        if(df_train['image_name'][ind]==name):
            gt,pred=df_train['GT_coordinates'][ind],df_train['prediction_coordinates'][ind]
    for ind in df_val.index:
        if(df_val['image_name'][ind]==name):
            gt,pred=df_val['GT_coordinates'][ind],df_val['prediction_coordinates'][ind]
    if(gt!=[] and gt!='Nan'):
        gt=convert_to_list(gt)
        x1=gt[0],gt[1]
        y1=gt[2],gt[3]
    if(pred!=[] and pred!='Nan'):
        pred=convert_to_list(pred)
        x2=pred[0],pred[1]
        y2=pred[2],pred[3]

    return x1,y1,x2,y2 #GT:(x1,y1)-->(xmin,ymin,xmax,ymax) Prediction:(x2,y2)-->(xmin,ymin,xmax,ymax)

def get_iou(name):
    df_train = pd.read_csv(chemins.train_csv)
    df_val=pd.read_csv(chemins.val_csv)
    iou='Nan'
    for ind in df_train.index:
        if(df_train['image_name'][ind]==name):
            iou=str(df_train['IOU'][ind])

    for ind in df_val.index:
        if(df_val['image_name'][ind]==name):
            iou=str(df_train['IOU'][ind])
    return iou


def display_image(image_name):

    path="/home/thedoctor/Téléchargements/PIA2/Algeria Cars Final/"+image_name
    iou='IoU='+get_iou(image_name)
    img = cv2.imread(path)
    #img = img[80:900, 1:900]
    if iou!='IoU=Nan':
        x1,y1,x2,y2=get_gt_pred(image_name) #GT:(x1,y1)-->(xmin,ymin,xmax,ymax) Prediction:(x2,y2)-->(xmin,ymin,xmax,ymax)
        cv2.rectangle(img, x1, y1, (0, 255, 0),4)# GT ['683', '560', '802', '660']
        cv2.putText(img,'GT',(y1[0],x1[1]),cv2.FONT_ITALIC,0.75,(0, 255, 0),1,cv2.LINE_AA)  
        cv2.rectangle(img, x2, y2, (255, 0, 0),4) # Prediction 682.3739013671875, 566.7169799804688, 802.5345458984375, 664.6019897460938
        cv2.putText(img,'Prediction',(y2[0],y2[1]+5),cv2.FONT_ITALIC,0.75,(255,0,0),1,cv2.LINE_AA)
        if x1[1]<x2[1]:
            cv2.putText(img,iou,(x1[0],x1[1]-5),cv2.FONT_ITALIC,0.75,(255,0,0),1,cv2.LINE_AA) #x1[0] is the x and x1[1] is the y
        else:
            cv2.putText(img,iou,(x2[0],x2[1]-5),cv2.FONT_ITALIC,0.75,(255,0,0),1,cv2.LINE_AA)
        #plt.savefig("car_1_V2.png")
        plt.imshow(img)
        plt.show()
        st.image(img)
    else: 
        st.warning('Image has no plate')
        plt.imshow(img)
        plt.show()
        st.image(img)
########################## SUMMARIZE CSV ##########################
def csv_summary(df):
    with st.expander("Display Images Predictions"):
        indexes=[i for i in df.index]
        indexes=tuple(indexes)
        selected_index=st.selectbox('select an index',indexes)
        image_name=df['image_name'][selected_index]
        display_image(image_name)
        
    index = df.index
    number_of_rows = len(index)
    st.write("Total number of cars: ",number_of_rows)

    df_temp=df.loc[df['GT_coordinates'] == 'Nan']
    index = df_temp.index
    number_of_rows = len(index)
    st.write("N°Cars with No Ground Truth: ",number_of_rows)
    with st.expander("show corresponding dataframe"):
        #st.write(df_temp)
        st.dataframe(df_temp)

    df_temp=df.loc[df['prediction_coordinates'] == 'Nan']
    index = df_temp.index
    number_of_rows = len(index)
    st.write("N°Cars with No Predicted Plate: ",number_of_rows)
    with st.expander("show corresponding dataframe"):
        st.write(df_temp)

    df_temp=df.loc[(df['GT_coordinates'] != 'Nan') & (df['prediction_coordinates'] != 'Nan')]
    index = df_temp.index
    number_of_rows = len(index)
    st.write("N°Cars with Ground Truth & Predicted Plate: ",number_of_rows)
    with st.expander("show corresponding dataframe"):
        st.write(df_temp)
########################## CONFUSION MATRIX ##########################
def confusion_matrix(iou,csv_path):
    '''returns tp,fp,tn,fn'''
    df = pd.read_csv(csv_path)
    tp=0
    fp=0
    tn=0
    fn=0
    total=0
    for ind in df.index:
        if(float(df['IOU'][ind])>=float(iou)):
            tp+=1
            total+=1
        elif(float(df['IOU'][ind])<float(iou)):
            fp+=1
            total+=1
        elif(df['GT_coordinates'][ind]=='Nan' and df['prediction_coordinates'][ind]=='Nan'):
            tn+=1
            total+=1
        else:
            fn+=1
            total+=1
    return tp,fp,tn,fn


########################## TRUE POSITIVES PLOT ##########################

# %matplotlib inline
def plot_all_iou(train_csv_path,val_csv_path):
    
    xpoints=[x/10 for x in range(1,11)]
    train_ypoints=[]
    val_ypoints=[]
    
    #fill the train_ypoints
    for iou in xpoints:
        tp,fp,tn,fn=confusion_matrix(iou,train_csv_path)
        tp=100*(tp/900)
        train_ypoints.append(tp)
        
    #fill the val_ypoints
    for iou in xpoints:
        tp,fp,tn,fn=confusion_matrix(iou,val_csv_path)
        tp=100*(tp/100)
        val_ypoints.append(tp)
    
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12, 5), constrained_layout=True, sharey=True)
    ax1.plot(xpoints, train_ypoints,color='green')
    ax1.set_title('Train Data (total=900)')
    ax1.set_xlabel('IoU')
    ax1.set_ylabel('% Of True Positives',fontsize=13)

    ax2.plot(xpoints, val_ypoints,color='orange')
    ax2.set_xlabel('IoU')
    ax2.set_title('Validation Data (total=100)')
    
    fig.suptitle('True Positives for Train, Validation Data', fontsize=16)
    st.pyplot(fig)


########################## PRECISION PLOT ##########################
def get_precision(iou,prediction_csv):
    tp,fp,tn,fn=confusion_matrix(iou,prediction_csv)
    precision=float(tp/(tp+fp))
    return precision

#%matplotlib inline
def plot_all_precision(train_csv_path,val_csv_path):
    
    xpoints=[x/10 for x in range(1,11)]
    train_ypoints=[]
    val_ypoints=[]
    
    #fill the train_ypoints
    for iou in xpoints:
        precision=get_precision(iou,train_csv_path)
        train_ypoints.append(precision)
    
    #fill the val_ypoints
    for iou in xpoints:
        precision=get_precision(iou,val_csv_path)
        val_ypoints.append(precision)

    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12, 5), constrained_layout=True, sharey=True)
    ax1.plot(xpoints, train_ypoints,color='green')
    ax1.set_title('Train Data (total=900)')
    ax1.set_xlabel('IoU')
    ax1.set_ylabel('Precision Values',fontsize=12)

    ax2.plot(xpoints, val_ypoints,color='orange')
    ax2.set_xlabel('IoU')
    ax2.set_title('Validation Data (total=100)')

    fig.suptitle('Precision for Train, Validation Data', fontsize=16)
    st.pyplot(fig)

########################## RECALL PLOTS ##########################
def get_recall(iou,prediction_csv):
    tp,fp,tn,fn=confusion_matrix(iou,prediction_csv)
    recall=float(tp/(tp+fn))
    return recall

#%matplotlib inline
def plot_all_recall(train_csv_path,val_csv_path):
    
    xpoints=[x/10 for x in range(1,11)]
    train_ypoints=[]
    val_ypoints=[]
    
    #fill the train_ypoints
    for iou in xpoints:
        recall=get_recall(iou,train_csv_path)
        train_ypoints.append(recall)
    
    #fill the val_ypoints
    for iou in xpoints:
        recall=get_recall(iou,val_csv_path)
        val_ypoints.append(recall)

    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12, 5), constrained_layout=True, sharey=True)
    ax1.plot(xpoints, train_ypoints,color='green')
    ax1.set_title('Train Data (total=900)')
    ax1.set_xlabel('IoU')
    ax1.set_ylabel('Recall Values',fontsize=12)

    ax2.plot(xpoints, val_ypoints,color='orange')
    ax2.set_xlabel('IoU')
    ax2.set_title('Validation Data (total=100)')

    fig.suptitle('Recall for Train, Validation Data', fontsize=16)
    st.pyplot(fig)

########################## F1SCORE PLOTS ##########################
def get_f1score(iou,prediction_csv):
    recall=get_recall(iou,prediction_csv)
    precision=get_precision(iou,prediction_csv)
    if(recall==0.0 and precision==0.0):
        f1score=0
    else:
        f1score=2.0*(float(recall*precision)/float(recall+precision))
    return f1score

#%matplotlib inline
def plot_all_f1score(train_csv_path,val_csv_path):
    
    xpoints=[x/10 for x in range(1,11)]
    train_ypoints=[]
    val_ypoints=[]
    
    #fill the train_ypoints
    for iou in xpoints:
        f1score=get_f1score(iou,train_csv_path)
        train_ypoints.append(f1score)
    
    #fill the val_ypoints
    for iou in xpoints:
        f1score=get_f1score(iou,val_csv_path)
        val_ypoints.append(f1score)
    
    #plot_all_f1score('train_predictions_27_02_2022.csv')
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12, 5), constrained_layout=True, sharey=True)
    ax1.plot(xpoints, train_ypoints,color='green')
    ax1.set_title('Train Data (total=900)')
    ax1.set_xlabel('IoU')
    ax1.set_ylabel('F1score Values',fontsize=12)

    ax2.plot(xpoints, val_ypoints,color='orange')
    ax2.set_xlabel('IoU')
    ax2.set_title('Validation Data (total=100)')
    
    fig.suptitle('F1score for Train, Validation Data', fontsize=16)
    st.pyplot(fig)
