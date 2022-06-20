# CSCI-566 Assignment 1

## The objectives of this assignment
* Implement the forward and backward passes as well as the neural network training procedure
* Implement the widely-used optimizers and training tricks including dropout
* Implement a convolutional neural network for image classification from scratch

## Work on the assignment
Working on the assignment in a virtual environment is highly encouraged.
In this assignment, please use Python `3.6`.
You will need to make sure that your virtualenv setup is of the correct version of python.

Please see below for executing a virtual environment.
```shell
cd csci566-assignment1
pip3 install virtualenv # If you didn't install it

# replace your_virtual_env with the virtual env name you want
virtualenv -p $(which python3) your_virtual_env
source your_virtual_env/bin/activate

# install dependencies
pip3 install -r requirements.txt

# work on the assignment
deactivate # Exit the virtual environment
```

## Work with IPython Notebook
To start working on the assignment, simply run the following command to start an ipython kernel.
```shell
# add your virtual environment to jupyter notebook
source your_virtual_env/bin/activate
python -m ipykernel install --user --name=your_virtual_env

# port is only needed if you want to work on more than one notebooks
jupyter notebook --port=your_port_number

```
and then work on each problem with their corresponding `.ipynb` notebooks.
Check the python environment you are using on the top right corner.
If the name of environment doesn't match, change it to your virtual environment in "Kernel>Change kernel".

## Problems
In each of the notebook file, we indicate `TODO` or `Your Code` for you to fill in with your implementation.
Majority of implementations will also be required under `lib` with specified tags.

### Problem 1: Basics of Neural Networks
The IPython Notebook `Problem_1.ipynb` will walk you through implementing the basics of neural networks.

### Problem 2: CNNs for Image Classification
The IPython Notebook `Problem_2.ipynb` will walk you through implementing a convolutional neural network (CNN) from scratch and using it for image classification.





