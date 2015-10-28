function  tvg_matcaffe_init(use_gpu, model_def_file, model_file)
% matcaffe_init(model_def_file, model_file, use_gpu)
% Initilize matcaffe wrapper

if nargin < 1
  error('Missing argument use_gpu');
end

if nargin < 2 || isempty(model_def_file)
  error('Missing argument model_def_file');
end

if nargin < 3 || isempty(model_file)
  error('Missing argument model_file');
end

% set to use GPU or CPU
if use_gpu
  fprintf('Using GPU Mode\n');
  caffe('set_mode_gpu');
else
  fprintf('Using CPU Mode\n');
  caffe('set_mode_cpu');
end
fprintf('Done with set_mode\n');


if caffe('is_initialized') == 0
  if exist(model_file, 'file') ~= 2
    error('You need a network model file');
  end
  if exist(model_def_file,'file') ~= 2
    error('You need the network prototxt definition');
  end
  caffe('init', model_def_file, model_file)
end
fprintf('Done with init\n');

% put into test mode
caffe('set_phase_test');
fprintf('Done with set_phase_test\n');
