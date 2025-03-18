import cv2
import os
import matplotlib.pyplot as plt
import  numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import pickle
import streamlit as st
from skimage.feature import hog
from skimage.feature import local_binary_pattern
import torch
from Nueral_Network import NueralNetwork
from CNN import MaskPredictorCNN

class Predict():
    
    def __init___():
        pass


if __name__=="__main__":

    st.header("Mask Image Classification")
    st.markdown("""
    This website focuses on classifying images with masks.  

    The following methods are used for preprocessing:  
    1. Histogram of Gradients
    2. Local Binary Pattern
                
    The following models are used to classify
    1. SVC
    2. Nueral Network
    3. CNN
    """)

    img_file = st.file_uploader("Upload an image",type=["jpg", "jpeg", "png"])

    if img_file is not None:

        st.image(img_file,"You uploaded the following image")

        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,dsize=(128,128))

        st.image(img,"Resize and convert to grayscale")

        img_norm = img/255.0

        with st.spinner("Wait for it...", show_time=True):

        
            st.subheader("Using Histogram Of Gradient")
            st.image("image/HOG_SVC_Results.png","testing result")

            features, h_img = hog(img, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True)

            st.image(h_img/255.0,"Step 2: HOG image")

            svc_hog = pickle.load(open("model/HOG_SVC.pkl","rb"))

            y_pred =svc_hog.predict([features])

            if y_pred==0: res_hog_svc = "without mask"
            else: res_hog_svc = "with mask"

            st.write(f"The person is :green[{res_hog_svc}]")


            st.subheader("Using Local Binary Pattern")
            st.image("image/LBP_SVC_Results.png","testing result")

            features_lbp = local_binary_pattern(img,P=8, R=1, method="uniform").flatten()

            svc_lbp = pickle.load(open("model/LBP_SVC.pkl","rb"))

            y_pred_lbp =svc_lbp.predict([features_lbp])

            if y_pred_lbp==0: res_lbp_svc = "without mask"
            else: res_lbp_svc = "with mask"

            st.write(f"The person is :green[{res_lbp_svc}]")

            st.subheader("Nueral Network with HOG Feature input")
            st.image("image/HOG_with_NN.png","testing result")

            features_tensor = torch.tensor(features,dtype=torch.float32)

            nnet = NueralNetwork()
            nnet.load_state_dict(torch.load("model/NueralNetwork",weights_only=True))
            nnet.eval()

            with torch.inference_mode():
                y_pred_nn = nnet(features_tensor)
                y_pred_nn = torch.round(torch.sigmoid(y_pred_nn))
            
            if y_pred_nn==0: res_nn = "without mask"
            else: res_nn = "with mask"

            st.write(f"The person is :green[{res_nn}]")

            st.subheader("CNN")
            st.image("image/CNN_results.png","test results")

            mpcnn = MaskPredictorCNN(1,32,1,128)
            nnet.load_state_dict(torch.load("model/CNN",weights_only=True))

            img_cnn = torch.tensor(img,dtype=torch.float32)
            img_cnn = img_cnn.unsqueeze(dim=0).unsqueeze(dim=0)


            mpcnn.eval()
            l1,l2,y_logits_test = mpcnn(img_cnn)
            y_pred_cnn = torch.round(torch.sigmoid(y_logits_test))

            st.write("Feature Maps layer 1")

            feature_maps_l1 = l1.squeeze(0).detach().numpy()
            rows, cols = 4, 8
            fig, axes = plt.subplots(rows, cols, figsize=(12, 6))

            for i, ax in enumerate(axes.flat):
                if i < feature_maps_l1.shape[0]:
                    ax.imshow(feature_maps_l1[i], cmap="gray")  # Display feature map
                    ax.axis("off")  # Hide axis

            st.pyplot(fig)



            st.write("Feature Maps layer 2")
            feature_maps_l2 = l2.squeeze(0).detach().numpy()
            rows, cols = 4, 8

            fig, axes = plt.subplots(rows, cols, figsize=(12, 6))

            for i, ax in enumerate(axes.flat):
                if i < feature_maps_l2.shape[0]:
                    ax.imshow(feature_maps_l2[i], cmap="gray")  # Display feature map
                    ax.axis("off")  # Hide axis

            st.pyplot(fig)
            
            if y_pred_cnn==0: res_cnn = "without mask"
            else:  res_cnn = "with mask"

            st.write(f"The person is :green[{res_nn}]")        
        
        st.success("Done")

