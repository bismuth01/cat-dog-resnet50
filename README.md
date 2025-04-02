# ResNet50 model on Cats VS Dogs dataset
ResNet50 model implemented from scratch using high level keras layers. Bottle neck architecture used for faster training times.

## Data preprocessing
<a href="https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset">Cats VS Dogs dataset</a> was used for 80 % training dataset and 20 % validation dataset while various data augmentation methods were applied. <a href="https://www.kaggle.com/datasets/aleemaparakatta/cats-and-dogs-mini-dataset">Cats and Dogs mini dataset</a> was used for testing dataset.

All images were resized to 128 x 128 pixels and a batch size of 32 was decided on.

## Building the model
A generic function of bottle neck architecture was designed to easily set up layers. Categorical dataset type and classification was chosen to make the model useful for any kind of image classification problem.
- Total params: 9,170,818
- Trainable params: 9,148,418
- Non-trainable params: 22,400

## Training process
Trained for 50 Epochs with Early Stop callback monitoring validation loss and patience of 15 Epochs, and Redue Learing Rate on Plateau callback monitoring validation loss, patience of 5 Epochs and reduction factor of 0.5.

## Training results
Metrics acheived: -
- Training accuracy: 97.97 %
- Validation accuracy: 93.64 %
- Testing accuracy: 98.69 %