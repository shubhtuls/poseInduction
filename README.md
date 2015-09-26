# Pose Induction.

[Shubham Tulsiani](http://cs.berkeley.edu/~shubhtuls), [Joao Carreira](http://www.cs.berkeley.edu/~carreira/) and [Jitendra Malik](http://cs.berkeley.edu/~malik). Pose Induction for Novel Object Categories. In ICCV, 2015.

### 0) Setup
- Download the code
```git clone --recursive https://github.com/shubhtuls/poseInduction```

- We first need to download the required datasets (PASCAL VOC and PASCAL3D+) and additional annotations. In addition, we also need to reorganize some data. To do this automatically, run
```bash initSetup.sh```

- Edit the required paths in 'startup.m', specially if you've used a local copy of some data instead of downloading via initSetup.sh

- Compile external/caffe (this is a slightly modified and outdated version of the [original](http://caffe.berkeleyvision.org/)). Sample compilation instructions are provided below. In case of any issues, refer to the installation instructions on the [caffe website](http://caffe.berkeleyvision.org/).

```
cd external/caffe
cp Makefile.config.example Makefile.config
make -j 8
#edit MATLAB_DIR in Makefile.config
make matcaffe pycaffe
cd ../..
```

### 1) Demo
- Initialize matlab in the root directory of the code.

- Run
``` startup; demo(); ```.
this will download our pretrained model and demonstrate predicted pose for a few images. Note that all the object classes in the demo images are novel (except for car which serves as a sanity check).

### 2) Training Models and Reproducing Experiments
This part of the codebase is still under construction. We'll update the instructions shortly.

