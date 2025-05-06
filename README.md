# USC8 AI Workshop

This repository contains the educational material (Jupyter notebooks) prepared for the Artificial Intelligence workshop organized by USC8. The notebooks cover various applications of Machine Learning and Deep Learning in astronomy.

## Contents

The repository includes the following main notebooks:

*   **`gmnist_classifier_light.ipynb`**:
    *   **Objective**: Classify images of galaxies based on their morphology (4 classes: smooth_round, smooth_cigar, edge_on_disk, unbarred_spiral).
    *   **Dataset**: Uses the Galaxy MNIST dataset (PNG images in 3 optical bands *grz* from DECaLS/Galaxy Zoo).
    *   **Techniques**: Implements Convolutional Neural Network (CNN) classifiers using PyTorch and Torchvision. Shows how to use a pre-trained architecture (ResNet) and how to define a custom CNN. Includes data augmentation, model evaluation, feature map visualization, and activation map visualization (Grad-CAM).
    *   **Main Libraries**: `pytorch`, `torchvision`, `pandas`, `numpy`, `matplotlib`, `PIL`, `opencv-python`, `pytorch-grad-cam`.

*   **`redshift_regression.ipynb`**:
    *   **Objective**: Predict the photometric redshift (`z_phot`) of astronomical objects.
    *   **Dataset**: Uses tabular data (magnitudes, errors, radii, etc.) from an SDSS (Sloan Digital Sky Survey) query, loaded from a FITS file (`catania_cavuoti.fit`).
    *   **Techniques**: Explores different regression algorithms from Scikit-learn (Random Forest, K-Nearest Neighbors, Multi-Layer Perceptron). Includes data pre-processing (scaling), feature importance analysis (with Random Forest), and hyperparameter optimization (GridSearch).
    *   **Main Libraries**: `scikit-learn`, `pandas`, `numpy`, `astropy`, `matplotlib`, `seaborn`.

*   **`time_series_classification.ipynb`**:
    *   **Objective**: Classify *synthetic* astronomical time series (light curves) into 7 categories (Cepheid, RR Lyrae, Eclipsing Binary, Delta Scuti, LPV, Flare Star, Rotational Modulation).
    *   **Dataset**: The data is generated *within* the notebook itself using functions that simulate different classes of stellar variability.
    *   **Techniques**: Uses Recurrent Neural Networks (RNNs, specifically LSTM and GRU) with PyTorch. Focuses on building and improving RNN architectures (introducing multiple layers, bidirectionality, dropout) starting from basic models.
    *   **Main Libraries**: `pytorch`, `numpy`, `matplotlib`, `scikit-learn` (for splitting), `torchmetrics`, `seaborn`.

Other files:
*   **`.gitignore`**: Standard file indicating which files or folders Git should ignore.
*   **`LICENSE`**: File specifying the license under which the material is distributed (MIT License).

## How to Run Notebooks in Google Colab (Recommended Method)

The easiest way to interact with these notebooks is using Google Colaboratory (Colab), a free service that allows you to run Python code directly in your browser, without needing local installations and providing access to computational resources (like free GPUs, especially useful for the PyTorch-based notebooks).

**Option 1: Direct Links ("Open in Colab" Badges)**

Click the badges below to open each notebook directly in Google Colab:

*   **Redshift Regression:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simoneriggi/usc8-ai-workshop/blob/main/redshift_regression.ipynb)
*   **Galaxy MNIST Classifier:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simoneriggi/usc8-ai-workshop/blob/main/gmnist_classifier_light.ipynb)
*   **Time Series Classification:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simoneriggi/usc8-ai-workshop/blob/main/time_series_classification.ipynb)

**Option 2: Manual Opening from Colab**

1.  Open [Google Colab](https://colab.research.google.com/).
2.  Go to `File` -> `Open notebook`.
3.  Select the `GitHub` tab.
4.  Paste the URL of this repository: `https://github.com/simoneriggi/usc8-ai-workshop`
5.  Press Enter or click the search icon.
6.  The list of files should appear. Click on the `.ipynb` notebook you wish to open.

Once opened in Colab, you can execute the code cells one by one by pressing `Shift + Enter` or using the buttons in the interface. The notebooks contain cells to install necessary libraries (`%pip install ...` or `!pip install ...`), so they should work directly in Colab.

Remember that the Colab environment is temporary; if you make changes you want to save, make sure to do so (`File -> Save a copy in Drive` or `File -> Download .ipynb`).

## Prerequisites

*   To use Google Colab: A Google account.
*   A basic understanding of Python and fundamental Machine Learning concepts is helpful for following the notebooks. Specific libraries are installed directly within the notebooks themselves.

## License

The content of this repository is released under the MIT License. For more details, see the `LICENSE` file.

---

We hope this workshop is useful and interesting!