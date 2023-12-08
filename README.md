# Crop Type Classification by DESIS Hyperspectral Imagery and Machine Learning Algorithms

This repository provides the implementation of a novel Wavelet-based Convolutional Neural Network (Wavelet CNN) framework for the classification of Hyperspectral Images (HSIs). The framework integrates wavelet transforms with an attention mechanism (AM) and convolutional neural networks (CNNs) to extract and utilize essential features from HSIs.

## Methodology

The proposed approach takes advantage of wavelet analysis and attention mechanisms to enhance feature extraction from HSIs. The methodology involves several key stages:

1. **Dimensional Reduction**: The HSI, initially of dimensions W×H×M (width, height, and spectral bands, respectively), undergoes dimensional reduction through factor analysis. The resulting output has reduced spectral dimensions W×H×C, where C is the number of selected features (reduced bands).

2. **Patch Extraction**: The dimensionally-reduced data is partitioned into 3-D patches of size (NS×D×D×C), where NS represents the number of samples and D×D is the patch window size.

3. **Wavelet Transform**: Each patch is transformed using the Haar wavelet to generate four sub-band components: fHH, fLL, fLH, and fHL, which decompose the input patch as follows:

    ```
    fHH = [ 1 -1  -1  1]
    fLL = [ 1  1   1  1]
    fLH = [-1  1  -1  1]
    fHL = [-1 -1   1  1]
    ```

4. **CNN Feature Extraction**: The wavelet sub-bands are passed through CNN layers to extract spatial and spectral features.

5. **Attention Mechanism**: Spectral AMs are integrated into the network to prioritize significant features, inspired by human visual attention systems that focus on pertinent and localized information.

6. **Feature Concatenation**: The outputs from different levels of decomposition are concatenated to preserve a rich set of features.

7. **Convolution and Pooling Operations**: Strides and mean-pooling are applied to reduce feature dimensions and mitigate overfitting.

8. **Regularization and Activation**: Dropout layers and ReLU activation functions are employed along with batch normalization to enhance model generalization.

9. **Classification**: A fully connected layer with a SoftMax activation function assigns a probability to each class, with the highest value indicating the predicted class.

10. **Loss Evaluation**: Cross-entropy loss is used to measure the model's performance and guide the optimization of predictions for new datasets.

The attention mechanism is specifically designed to capture correlations and enhance the representation of relevant features, which is crucial for accurate HSI classification.

## Code Structure

The code provided in this repository outlines the steps to construct and train the Wavelet CNN model according to the methodology described. It encompasses wavelet transform functions, attention modules, and CNN architecture tailored for effective HSI processing.

## Dependencies

The following libraries are required for the implementation:

- Keras
- TensorFlow
- NumPy
- Matplotlib
- Scipy
- Scikit-learn
- Spectral (Installation command: `!pip install spectral`, if using Jupyter notebook)
## Usage

To utilize this model, make sure all dependencies are installed. The model can be instantiated and compiled using the provided functions and classes.

```python
from keras import backend as Kb
from tensorflow.keras.models import Model

# Initialize the model
model = get_wavelet_cnn_model()

# Compile the model with the desired optimizer, loss function, and metrics
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
Train the model with your HSI dataset as follows:

# Fit the model on the training data
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
```

## Paper Reference
For a detailed explanation of the model and methodologies, please refer to the following paper:

[the documentation ](https://ieeexplore.ieee.org/abstract/document/10032208)

## Contact
For any queries or discussions regarding the Attention Wavelet CNN model, please open an issue in this repository or contact the maintainers directly.

## Acknowledgments
Credit to the authors and contributors of the research paper that inspired this implementation.

## Hashtags

#WaveletCNN
#HyperspectralImaging
#DeepLearning
#RemoteSensing
#FeatureExtraction
#AttentionMechanism
#ImageClassification
#CropYieldPrediction 
