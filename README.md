Note - this codebase is still under construction. We'll update the instructions and release trained models shortly.

# Pose Induction.

[Shubham Tulsiani](http://cs.berkeley.edu/~shubhtuls), [Joao Carreira](http://www.cs.berkeley.edu/~carreira/) and [Jitendra Malik](http://cs.berkeley.edu/~malik). Pose Induction for Novel Object Categories. In ICCV, 2015.

### 0) Setup
- Download the code
```git clone --recursive https://github.com/shubhtuls/caffePose```

- We first need to download the required datasets (PASCAL VOC and PASCAL3D+) and additional annotations. In addition, we also need to reorganize some data. To do this automatically, run

```bash initSetup.sh```

(if PASCAL VOC server is not working, uncomment the corresponding lines from the script and move a local copy to the desired location)

- Edit the required paths in 'startup.m', specially if you've used a local copy of some data instead of downloading via initSetup.sh

- Compile external/caffe
