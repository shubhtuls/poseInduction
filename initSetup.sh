#directory for data
mkdir data

# Download PASCAL 3D
wget ftp://cs.stanford.edu/cs/cvgl/PASCAL3D+_release1.1.zip
unzip PASCAL3D+_release1.1.zip
mv PASCAL3D+_release1.1 PASCAL3D
mv PASCAL3D+* data/
mv PASCAL3D data/

# move all imagenet images in PASCAL3D+ in one folder, to resemble the pascal VOC setup
mkdir -p data/imagenet/images
for x in $(ls data/PASCAL3D/Images | grep imagenet); do mv data/PASCAL3D/Images/$x/*.JPEG data/imagenet/images/; done

# Download keypoint annotations
mkdir ./data/segkps
wget -P ./data/segkps/ http://www.cs.berkeley.edu/~shubhtuls/cachedir/vpsKps/segkps.zip
unzip ./data/segkps/segkps.zip -d ./data/segkps/
wget -P ./data/  http://www.cs.berkeley.edu/~shubhtuls/cachedir/poseInduction/vocKpMetadata.mat
# Download PASCAL VOC
wget http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
tar -xf VOCtrainval_11-May-2012.tar
mv VOCdevkit data/
mv VOCtrainval_11-May-2012.tar data/

tar -xf VOCdevkit_18-May-2011.tar
mv VOCdevkit/* ./data/VOCdevkit/
rm -r VOCdevkit

# Download and install caffe
mkdir external
dirVar="./external/caffe"
if [ ! -d "$dirVar" ]; then
    # Control will enter here if $DIRECTORY doesn't exist.
    git clone git@bitbucket.org:shubhtuls/caffe.git $dirVar
    touch "$dirVar/__init__.py"
    cd "$dirVar"
    cp Makefile.config.example Makefile.config
    git checkout pose
    git pull origin pose
    #now compile caffe and matcaffe in external/caffePose
    make -j8 caffe
    make matcaffe
    cd ../..
    #echo $dirVar
fi
