# CRF-RNN for Semantic Image Segmentation
![sample](sample.png)

<b>Live demo:</b> [http://crfasrnn.torr.vision](http://crfasrnn.torr.vision)

This package contains code for the "CRF-RNN" semantic image segmentation method, published in the ICCV 2015 paper [Conditional Random Fields as Recurrent Neural Networks](http://www.robots.ox.ac.uk/~szheng/papers/CRFasRNN.pdf). This paper was initially described in an [arXiv tech report](http://arxiv.org/abs/1502.03240). Our software is built on top of the [Caffe](http://caffe.berkeleyvision.org/) deep learning library. The current version was developed by:

[Sadeep Jayasumana](http://www.robots.ox.ac.uk/~sadeep/),
[Shuai Zheng](http://kylezheng.org/),
[Bernardino Romera Paredes](http://romera-paredes.com/), and
[Zhizhong Su](suzhizhong@baidu.com).

Supervisor: [Philip Torr](http://www.robots.ox.ac.uk/~tvg/)

Our work allows computers to recognize objects in images, what is distinctive about our work is that we also recover the 2D outline of the object.

Currently we have trained this model to recognize 20 classes. This software allows you to test our algorithm on your own images – have a try and see if you can fool it, if you get some good examples you can send them to us.

Why are we doing this? This work is part of a project to build augmented reality glasses for the partially sighted. Please read about it here: [smart-specs](http://www.va-st.com/smart-specs/). 

For demo and more information about CRF-RNN please visit the project website <http://crfasrnn.torr.vision>.

If you use this code/model for your research, please consider citing the following paper:
```
@inproceedings{crfasrnn_ICCV2015,
    author = {Shuai Zheng and Sadeep Jayasumana and Bernardino Romera-Paredes and Vibhav Vineet and Zhizhong Su and Dalong Du and Chang Huang and Philip H. S. Torr},
    title  = {Conditional Random Fields as Recurrent Neural Networks},
    booktitle = {International Conference on Computer Vision (ICCV)},
    year   = {2015}
}
```


#Installation Guide

You need to compile the modified Caffe library in this repository. Instructions for Ubuntu 14.04 are included below. You can also consult the generic [Caffe installation guide](http://caffe.berkeleyvision.org/installation.html).


###1.1 Install dependencies
#####General dependencies
```
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install --no-install-recommends libboost-all-dev
```

#####CUDA 
Install CUDA correct driver and its SDK. Download CUDA SDK from Nvidia website. 

In Ubuntu 14.04. You need to make sure the required tools are installed. You might need to blacklist the required modules so that they do not interfere with the driver installation. You also need to uninstall your default Nvidia Driver first.
```
sudo apt-get install freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev
``` 
open /etc/modprobe.d/blacklist.conf and add:
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

When you restart your PC, before loging in, try "Ctrl+Alt+F1" switch to a text-based login. Try:
```
sudo service lightdm stop
chmod +x cuda*.run
sudo ./cuda*.run
```

#####BLAS
Install ATLAS or OpenBLAS or MKL.

#####Python 
Install Anaconda Python distribution or install the default Python distribution with numpy/scipy/...

#####MATLAB (optional)
Install MATLAB using a standard distribution.

###1.2 Build the custom Caffe version
Set the path correctly in the Makefile.config. You can copy the Makefile.config.example to Makefile.config, as most common parts are filled already. You need to change it according to your environment.

After this, in Ubuntu 14.04, try:
```
make
```

If there are no error messages, you can then compile and install the python and matlab wrappers:
```
make matcaffe
```

```
make pycaffe
```

That's it! Enjoy our software!


###1.3 Run the demo
Matlab and Python scripts for running the demo are available in the matlab-scripts and python-scripts directories, respectively. You can choose either of them. Note that you should change the paths in the scripts according your environment.

####For Python fans:
First you need to download the model. In Linux, this is:
```
sh download_trained_model.sh
```
Atlernatively, you can also get the model by directly clicking the link in python-scripts/README.md.

Get into the python-scripts folder, and then type:
```
python crfasrnn_demo.py
```
You will get an output.png image.

To use your own images, just replace "input.jpg" in the crfasrnn_demo.py file.

####For Matlab fans:
First you need to download the model. In Linux, this is:
```
sh download_trained_model.sh
```
Atlernatively, you can also get the model by directly clicking the link in matlab-scripts/README.md.

Get into the matlab-scripts folder, load your matlab, then run the crfrnn_demo.m.

To use your own images, just replace "input.jpg" in the crfrnn_demo.m file.

You can also find part of our model in [MatConvNet](http://www.vlfeat.org/matconvnet/pretrained/).

# GPU use
GPU mode is enable for the Meanfield layers. Please initialize your device before creating your network or it
might fail otherwise.

# LICENSE
CRF-RNN feature in Caffe is implemented for the paper:
Shuai Zheng, Sadeep Jayasumana, Bernardino Romera-Paredes, Vibhav Vineet, Zhizhong Su, Dalong Du, Chang Huang, Philip H. S. Torr.
Conditional Random Fields as Recurrent Neural Networks. IEEE ICCV 2015.

Shuai Zheng, Sadeep Jayasumana, Bernardino Romera-Paredes, and Philip H. S. Torr are with University of Oxford.
Vibhav Vineet did this work when he was with the University of Oxford, he is now with the Stanford University.
Zhizhong Su, Dalong Du, Chang Huang are with the Baidu Institute of Deep Learning (IDL).

CRF-RNN uses the Permutohedral lattice library, the DenseCRF library and the Caffe future version.

Permutohedral lattice library (BSD license) is from Andrew Adams, Jongmin Baek, Abe Davis. Fast High-Dimensional Filtering Using the
Permutohedral Lattice. Eurographics 2010.
DenseCRF library from Philipp Krahenbuhl and Vladlen Koltun. Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials.
NIPS 2011.

For more information about CRF-RNN please vist the project website http://crfasrnn.torr.vision.
