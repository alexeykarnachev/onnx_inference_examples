export CUDA_HOME=/usr/local/cuda/
export CUDNN_HOME=/usr/include/
apt install -y language-pack-en
locale-gen en_US.UTF-8
update-locale LANG=en_US.UTF-8
git clone https://github.com/microsoft/onnxruntime.git
./onnxruntime/build.sh --config Release --build_wheel --parallel --use_openmp --use_cuda --cudnn_home $CUDNN_HOME --cuda_home $CUDA_HOME
cp /workspace/onnxruntime/build/Linux/Release/dist/*.whl ./
