## CNN Fashion Model

This project implements a **Convolutional Neural Network (CNN)** using PyTorch and torchvision. The model is trained on the **FashionMNIST dataset**, which consists of images of clothing items.

## Dataset

The FashionMNIST dataset is used for training and testing the CNN. It contains 10 classes of clothing items:

- 0: T-shirt/top
- 1: Trouser
- 2: Pullover
- 3: Dress
- 4: Coat
- 5: Sandal
- 6: Shirt
- 7: Sneaker
- 8: Bag
- 9: Ankle boot

## Model Architecture

The CNN model consists of 2 convolutional blocks and a linear layer with 10 output classes. It is trained using the PyTorch library.

## Training and Testing

The model is trained and tested within the code. It includes training loops, evaluation, and prediction steps.

## Visualization

The code includes commented-out Matplotlib visualization code. Users can uncomment this code to visualize the changes in data at each step and every layer of the CNN.

## Usage

Clone the repository:

```bash
git clone https://github.com/Piyush2102020/Convolutional_Neural_network.git
cd Convolutional_Neural_network
python cnn.py
