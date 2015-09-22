function cnn_model = cnn_load_model(cnn_model, use_gpu)
% cnn_model = cnn_load_model(cnn_model, use_gpu)

if(nargin < 2 || use_gpu)
    caffe.set_mode_gpu()
else
    caffe.set_mode_cpu()
end
cnn_model.net = caffe.Net(cnn_model.definition_file, cnn_model.binary_file,'test');

end