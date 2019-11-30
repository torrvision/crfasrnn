# CRF-RNN for Semantic Image Segmentation

<b>Live demo:</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [http://crfasrnn.torr.vision](http://crfasrnn.torr.vision) <br/>
<b>PyTorch version:</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[http://github.com/sadeepj/crfasrnn_pytorch](http://github.com/sadeepj/crfasrnn_pytorch)<br/>
<b>Tensorflow/Keras version:</b> [http://github.com/sadeepj/crfasrnn_keras](http://github.com/sadeepj/crfasrnn_keras)<br/>

![sample](sample.png)

[![License (3-Clause BSD)](https://img.shields.io/badge/license-BSD%203--Clause-brightgreen.svg?style=flat-square)](https://github.com/torrvision/crfasrnn/blob/master/LICENSE)



This package contains code for the "CRF-RNN" semantic image segmentation method, published in the ICCV 2015 paper [Conditional Random Fields as Recurrent Neural Networks](http://www.robots.ox.ac.uk/~szheng/papers/CRFasRNN.pdf). This paper was initially described in an [arXiv tech report](http://arxiv.org/abs/1502.03240). The online demonstration based on this code won the Best Demo Prize at ICCV 2015. Our software is built on top of the [Caffe](http://caffe.berkeleyvision.org/) deep learning library. The current version was developed by:

[Sadeep Jayasumana](http://www.robots.ox.ac.uk/~sadeep/),
[Shuai Zheng](http://kylezheng.org/),
[Bernardino Romera Paredes](http://romera-paredes.com/), 
[Anurag Arnab](http://www.robots.ox.ac.uk/~aarnab/),
and
Zhizhong Su.

Supervisor: [Philip Torr](http://www.robots.ox.ac.uk/~tvg/)

Our work allows computers to recognize objects in images, what is distinctive about our work is that we also recover the 2D outline of objects. Currently we have trained this model to recognize 20 classes. This software allows you to test our algorithm on your own images – have a try and see if you can fool it, if you get some good examples you can send them to us.

Why are we doing this? This work is part of a project to build augmented reality glasses for the partially sighted. Please read about it here: [smart-specs](http://www.va-st.com/smart-specs/). 

For demo and more information about CRF-RNN please visit the project website: <http://crfasrnn.torr.vision>.

If you use this code/model for your research, please cite the following papers:
```
@inproceedings{crfasrnn_ICCV2015,
    author = {Shuai Zheng and Sadeep Jayasumana and Bernardino Romera-Paredes and Vibhav Vineet and
    Zhizhong Su and Dalong Du and Chang Huang and Philip H. S. Torr},
    title  = {Conditional Random Fields as Recurrent Neural Networks},
    booktitle = {International Conference on Computer Vision (ICCV)},
    year   = {2015}
}
```
```
@inproceedings{higherordercrf_ECCV2016,
	author = {Anurag Arnab and Sadeep Jayasumana and Shuai Zheng and Philip H. S. Torr},
	title  = {Higher Order Conditional Random Fields in Deep Neural Networks},
	booktitle = {European Conference on Computer Vision (ECCV)},
	year   = {2016}
}
```


## How to use the CRF-RNN layer
CRF-RNN has been developed as a custom Caffe layer named MultiStageMeanfieldLayer. Usage of this layer in the model definition prototxt file looks the following. Check the `matlab-scripts` or the `python-scripts` folder for more detailed examples.
```
# This is part of FCN, coarse is a blob coming from FCN
layer { type: 'Crop' name: 'crop' bottom: 'bigscore' bottom: 'data' top: 'coarse' }

# This layer is used to split the output of FCN into two. This is required by CRF-RNN.
layer { type: 'Split' name: 'splitting'
  bottom: 'coarse' top: 'unary' top: 'Q0'
}

layer {
  name: "inference1" # Keep the name "inference1" to load the trained parameters from our caffemodel.
  type: "MultiStageMeanfield" # Type of this layer
  bottom: "unary" # Unary input from FCN
  bottom: "Q0" # A copy of the unary input from FCN
  bottom: "data" # Input image
  top: "pred" # Output of CRF-RNN
  param {
    lr_mult: 10000 # learning rate for W_G
  }
  param {
    lr_mult: 10000 # learning rate for W_B
  }
  param {
    lr_mult: 1000 # learning rate for compatiblity transform matrix
  }
  multi_stage_meanfield_param {
    num_iterations: 10 # Number of iterations for CRF-RNN
    compatibility_mode: POTTS # Initialize the compatilibity transform matrix with a matrix whose diagonal is -1.
    threshold: 2
    theta_alpha: 160
    theta_beta: 3
    theta_gamma: 3
    spatial_filter_weight: 3
    bilateral_filter_weight: 5
  }
}
```
## Installation Guide
First, clone the project by running:
```
git clone --recursive https://github.com/torrvision/crfasrnn.git
```

You need to compile the modified Caffe library in this repository. Instructions for Ubuntu 14.04 are included below. You can also consult the generic [Caffe installation guide](http://caffe.berkeleyvision.org/installation.html) for further help.


### 1.1 Install dependencies
##### General dependencies
```
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install --no-install-recommends libboost-all-dev
```

##### CUDA (optional - needed only if you are planning to use a GPU for faster processing)
Install the correct CUDA driver and its SDK. Download CUDA SDK from Nvidia website. 

You might need to blacklist some modules so that they do not interfere with the driver installation. You also need to uninstall your default Nvidia Driver first.
```
sudo apt-get install freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev
``` 
open `/etc/modprobe.d/blacklist.conf` and add:
```
blacklist amd76x_edac
blacklist vga16fb
blacklist nouveau
blacklist rivafb
blacklist nvidiafb
blacklist rivatv
```
```
sudo apt-get remove --purge nvidia*
```

When you restart your PC, before logging in, try "Ctrl + Alt + F1" to switch to a text-based login. Try:
```
sudo service lightdm stop
chmod +x cuda*.run
sudo ./cuda*.run
```

##### BLAS
Install a BLAS library such as ATLAS, OpenBLAS or MKL. To install BLAS:
```
sudo apt-get install libatlas-base-dev 
```

##### Python 
Install Anaconda Python distribution or install the default Python distribution with numpy, scipy, etc.

##### MATLAB (optional - needed only if you are planning to use the MATLAB interface)
Install MATLAB using a standard distribution.

### 1.2 Build the custom Caffe version
Set the path correctly in the ``Makefile.config``. You can rename the ``Makefile.config.example`` to ``Makefile.config``, as most common parts are filled already. You may need to change it a bit according to your environment.

After this, in Ubuntu 14.04, try:
```
make
```

If there are no error messages, you can then compile and install the Python and Matlab wrappers:
To install the MATLAB wrapper (optional):
```
make matcaffe
```

To install the Python wrapper (optional):
```
make pycaffe
```

That's it! Enjoy our software!


### 1.3 Run the demo
MATLAB and Python scripts for running the demo are available in the ``matlab-scripts`` and ``python-scripts`` directories, respectively. Both of these scripts do the same thing - you can choose either.

#### Python users:
Change the directory to ``python-scripts``. First download the model that includes the trained weights. In Linux, this can be done by:
```
sh download_trained_model.sh
```
Alternatively, you can also get the model by directly clicking the link in ``python-scripts/README.md``.

To run the demo, execute:
```
python crfasrnn_demo.py
```
You will get an output.png image.

To use your own images, replace "input.jpg" in the ``crfasrnn_demo.py`` file.

#### MATLAB users:
Change the directory to ``matlab-scripts``. First download the model that includes the trained weights. In Linux, this can be done by:
```
sh download_trained_model.sh
```
Alternatively, you can also get the model by directly clicking the link in ``matlab-scripts/README.md``.

Load your MATLAB application and run ``crfrnn_demo.m``.

To use your own images, just replace "input.jpg" in the ``crfrnn_demo.m`` file.

You can also find a part of our model in [MatConvNet](http://www.vlfeat.org/matconvnet/pretrained/).


#### Explanation about the CRF-RNN layer:
If you would like to try out the CRF-RNN model we trained, you should keep the layer name as it is ("inference1"), so that the code will correctly load the parameters from the caffemodel. Otherwise, it will reinitialize parameters.

You should find out that the end-to-end trained CRF-RNN model does better than the alternatives. If you set the CRF-RNN layer name to "inference2", you should observe lower performance since the parameters for both CNN and CRF are not jointly optimized.


#### Training CRF-RNN on a new dataset:
If you would like to train CRF-RNN on other datasets, please follow the piecewise training described in our paper. In short, you should first train a strong pixel-wise CNN model. After this, you could plug our CRF-RNN layer into it by adding the MultiStageMeanfieldLayer to the prototxt file. You should then be able to train the CNN and CRF-RNN parts jointly end-to-end.

Notice that the current deploy.prototxt file we have provided is tailored for PASCAL VOC Challenge. This dataset contains 21 class labels including background. You should change the num_output in the corresponding layer if you would like to finetune our model for other datasets. Also, the deconvolution layer in current code does not allow initializing the parameters through prototxt. If you change the num_output there, you should manually re-initialize the parameters in the caffemodel file.

See ``examples/segmentationcrfasrnn`` for more information.


#### Why predictions are all black?
This could happen if you change layer names in the model definition prototxt, causing the weights not to load correctly. This could also happen if you change the number of outputs in deconvolution layer in the prototxt but not initialize the deconvolution layer properly. 

#### MultiStageMeanfield causes a segfault?
This error usually occurs when you do not place the ``spatial.par`` and ``bilateral.par`` files in the script path.

#### Python training script from third parties
We would like to thank martinkersner and MasazI for providing Python training scripts for CRF-RNN. 

1. [martinkersner's scripts](https://github.com/martinkersner/train-CRF-RNN)
2. [MasazI's scripts](https://github.com/MasazI/crfasrnn-training)

#### Merge with the upstream caffe
It is possible to integrate the CRF-RNN code into upstream Caffe. However, due to the change of the crop layer, the caffemodel we provided might require extra training to provide the same accuracy. mtourne kindly provided a version that merged the code with upstream caffe. 

1. [mtourne upstream version with CRFRNN](https://github.com/mtourne/crfasrnn)

#### GPU version of CRF-RNN
hyenal kindly provided a purely GPU version of CRF-RNN. This would lead to considerably faster training and testing.

1. [hyenal's GPU crf-rnn](https://github.com/hyenal/crfasrnn)

#### CRF-as-RNN as a layer in Lasagne
[Lasagne CRFasRNN layer](https://github.com/hapemask/crfrnn_layer)

#### Latest Caffe with CPU/GPU CRF-RNN
[crfasrnn-caffe](https://github.com/torrvision/caffe/tree/crfrnn)

#### Keras/Tensorflow version of CRF-RNN
[crfasrnn_keras](https://github.com/sadeepj/crfasrnn_keras)

Let us know if we have missed any other works from third parties.


For more information about CRF-RNN please visit the project website http://crfasrnn.torr.vision. Contact: <crfasrnn@gmail.com>
