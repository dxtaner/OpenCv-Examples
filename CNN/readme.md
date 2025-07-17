
✍️ MNIST Handwritten Digit Recognizer (Draw with Mouse)
=======================================================

This is a simple Python application that lets you draw a digit with your mouse, and uses a Convolutional Neural Network (CNN) trained on the MNIST dataset to recognize it.

🧠 Features
-----------

*   Draw digits (0–9) directly on a virtual canvas
*   Real-time prediction using a trained deep learning model
*   Clear and reset the canvas
*   Simple OpenCV + TensorFlow integration

🛠 Requirements
---------------

Install with:

    pip install tensorflow opencv-python numpy

📁 Files
--------

*   `mnist_cnn_model.h5`: The pre-trained CNN model trained on the MNIST dataset
*   `MnistPredict.py`: Main application to draw and predict digits
*   `MnistCnnTrain.py` _(optional)_: Script to train and save the model

▶️ How to Run
-------------

1.  Ensure `mnist_cnn_model.h5` is in the same folder.
2.  Run the app:
    
        python MnistPredict.py
    
3.  **Controls:**
    *   🖱️ Draw with left mouse button
    *   🔮 Press `p` to predict
    *   ♻️ Press `c` to clear
    *   ❌ Press `q` to quit

🧪 Example
----------


🧠 Model Info
-------------

The CNN architecture consists of:

*   2 Convolutional layers (ReLU + MaxPooling)
*   1 Dense hidden layer
*   Softmax output for 10 classes

To retrain:

    python MnistPredict.py

📜 License
----------

This project is open-source and free to use under the MIT License.

* * *

Created with ❤️ using OpenCV & TensorFlow
