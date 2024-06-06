# Medicinal Plant Identification using CNN-RNN Hybrid Model

This project focuses on developing a machine learning model to identify medicinal plants through image processing. The hybrid model combines Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) to leverage both spatial and sequential information for accurate plant identification.

## Project Overview

Medicinal plants play a crucial role in healthcare and traditional medicine. Accurate identification of these plants is essential for their proper utilization. This project utilizes deep learning techniques to create a robust model that can classify medicinal plants based on images.

### Key Features
- **CNN for Feature Extraction**: The Convolutional Neural Network (CNN) extracts meaningful features from input images, capturing spatial hierarchies and patterns.
- **RNN for Sequence Modeling**: The Recurrent Neural Network (RNN) processes the sequence of features extracted by the CNN, capturing temporal dependencies and patterns.
- **Data Preprocessing and Augmentation**: Includes normalization and data augmentation techniques to improve model generalization and performance.
- **End-to-End Training**: The model is trained end-to-end, allowing simultaneous learning of spatial and sequential patterns.
- **Evaluation and Visualization**: Comprehensive evaluation metrics and visualizations to assess model performance.

## Libraries and Dependencies

- **NumPy**: For numerical operations and array handling.
- **Pandas**: For data manipulation and analysis.
- **TensorFlow/Keras**: For building and training deep learning models.
- **Matplotlib**: For plotting and visualization.
- **scikit-learn**: For data splitting and evaluation metrics.
  **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/medicinal-plant-identification.git
   cd medicinal-plant-identification
  Run the Jupyter notebook for step-by-step instructions on training and evaluating the model:
jupyter notebook notebooks/Medicinal_Plant_Identification.ipynb 
