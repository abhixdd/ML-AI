# Image Classification Project - Abhinav A
## AIML Exit Examination

This repository contains an automated system to classify natural scene images (Forest, Glacier, Buildings, Sea, Mountain, Street) using Deep Learning.

### Repository Structure
- `ai_ml_mini_project.ipynb`: Comprehensive notebook containing:
    - Data exploration and distribution analysis.
    - Preprocessing and augmentation strategies.
    - CNN Model design and training logic.
    - Evaluation metrics and confusion matrix.
    - Error analysis and visualizations.
    - Answers to all 7 analytical questions.
- `app.py`: Premium Streamlit application for real-time scene classification.
- `cnn_intel_image_classification_model.keras`: Finalized Keras model used by the application.
- `README.md`: Execution instructions.

### Execution Instructions

#### 1. Environment Setup
Ensure you have Python 3.10+ installed. Install the required dependencies:
```bash
pip install torch torchvision keras numpy matplotlib seaborn streamlit pillow
```

#### 2. Running the Application
To launch the interactive scene classifier:
```bash
streamlit run app.py
```

#### 3. Notebook Analysis
Open `ai_ml_mini_project.ipynb` in any Jupyter environment to view the detailed model development and failure analysis.

### Model Features
- **Architecture**: Convolutional Neural Network (CNN) with batch normalization and dropout for regularization.
- **Backend**: Keras 3 (Multi-backend supported).
- **Target Size**: 150x150 pixels.
- **Categories**: 6 (Buildings, Forest, Glacier, Mountain, Sea, Street).

