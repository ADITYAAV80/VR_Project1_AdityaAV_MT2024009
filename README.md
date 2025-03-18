# Mask Image Classification

This project focuses on classifying images of people with and without masks using various machine learning and deep learning techniques.

## Features

- **Preprocessing Methods**:

  1. Histogram of Oriented Gradients (HOG)
  2. Local Binary Pattern (LBP)

- **Classification Models**:

  1. Support Vector Classifier (SVC)
  2. Neural Network
  3. Convolutional Neural Network (CNN)

- **Interactive Web Application**:
  - Built using Streamlit for uploading and classifying images.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo-url.git
   cd VR_MiniProject
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Download and unzip data.zip from [Onedrive](https://iiitbac-my.sharepoint.com/:u:/g/personal/aditya_av_iiitb_ac_in/EUxRRsZsmudDrIHz4nx7HesBm-Xj-yOyzlSWY4wY_2JI4g?e=chbdnw) and copy it in /Data

4. Download and pickle files from [Onedrive](https://iiitbac-my.sharepoint.com/:f:/g/personal/aditya_av_iiitb_ac_in/Epv7QqiitO5IoX1KtSW_jgsBhERum67geqYB_z5KHvFsaw?e=NuVLsj) and place it in /model

5. To run

   ```bash
   streamlit run app.py
   ```

6. Results

   1. HOG with SVC

      ![HOG Results](image/HOG_SVC_Results.png)

   2. LBP with SVC

      ![LBP Results](image/LBP_SVC_Results.png)

   3. Nueral network with HOG Input

      ![Nueral Network Results](image/HOG_with_NN.png)

   4. CNN Results

      ![CNN Results](image/CNN_results.png)
