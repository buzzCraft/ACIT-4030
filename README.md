https://arxiv.org/abs/1610.07584


In this assignment, the goal is to replicate the 3D GAN from Ref. [1].

The dataset to be used is ModelNet10, which can be downloaded here 

Download here 

The dataset is loaded using:

```python
import numpy as np
data = np.load("modelnet10.npz", allow_pickle=True)
train_voxel = data["train_voxel"] # Training 3D voxel samples
test_voxel = data["test_voxel"] # Test 3D voxel samples
train_labels = data["train_labels"] # Training labels (integers from 0 to 9)
test_labels = data["test_labels"] # Test labels (integers from 0 to 9)
class_map = data["class_map"] # Dictionary mapping the labels to their class names.
```

Visualizing a 3D voxel model can be done in matplotlib.pyplot like this:

```python
import matplotlib.pyplot as plt
ax = plt.figure().add_subplot(projection='3d')
index = <index to your model>
ax.voxels(<your voxel map,like train_voxel>[index])
plt.show()
```

Task A - Training the discriminator and generator

Train a 3D GAN using the dataset ModelNet10 as described in Ref. [1].  The size of the batch must be between 10 and 100 samples. The batch must be a random sample from the training dataset. You must train the 3D GAN with 3 or more epochs covering the full training dataset. These are the task requests:

    Show 5 samples from the training dataset.
    Show the total number of parameters of the two models.
    Save the trained model for later use.
    Present a figure of the history of the training loss of the discriminator and generator.
    Show 4 generated 3D voxel data using the trained generator model with different latent vector z.

Task B - Training the classifier

Use the trained and saved discriminator from Task A for classification as described in Ref. [1], but replace the linear SVM with a dense trainable layer with softmax activation function. The training must be performed with Adam Optimizer and cross-entropy loss function. The learning rate for the classifier can be 0.001. You must train the classifier with 2 or more epochs. After training, these are the task requests:

    Show the total number of parameters of the classifier.
    Plot in two different subplots the training loss and training accuracy history of the classifier.

The submission

A report and source code must be submitted that address all task requests of Task A and Task B. The report should not exceed 10 pages. Source code must have descriptive comments. Parts of open-source code from websites must be cited with source link, such as shown below:

```python
def function_from_web():
    # Function code from: [link]
    # or
    # Function code based on: [link]
    return
```