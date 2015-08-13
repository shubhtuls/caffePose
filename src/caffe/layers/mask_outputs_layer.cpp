#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/pose_layers.hpp"
#include <math.h>

namespace caffe {

template <typename Dtype>
void MaskOutputsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  kernel_size_ = this->layer_param_.convolution_param().kernel_size();
}

template <typename Dtype>
void MaskOutputsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> topShape(2);
  topShape[0] = bottom[0]->shape()[0];
  topShape[1] = kernel_size_;
  top[0]->Reshape(topShape);
}

template <typename Dtype>
void MaskOutputsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* start = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  const int num = bottom[0]->shape()[0];
  const int nchannels = bottom[0]->shape()[1];
  const int bottomOffset = nchannels;
  const int topOffset = kernel_size_;

  for (int n = 0; n < num; ++n) {
    int offsetIn = n*bottomOffset + start[n]*kernel_size_;
    int offsetOut = n*topOffset;
    //cout<<start[n];
    for (int c = 0; c < kernel_size_; ++c) {
      top_data[offsetOut+c] = bottom_data[offsetIn+c];
    }
  }
}


template <typename Dtype>
void MaskOutputsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* start = bottom[1]->cpu_data();

    const int count = bottom[0]->count();
    
    const int num = bottom[0]->shape()[0];
    const int nchannels = bottom[0]->shape()[1];

    const int bottomOffset = nchannels;
    const int topOffset = kernel_size_;

    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = Dtype(0);
    }

    for (int n = 0; n < num; ++n) {
        //LOG(INFO) << "Offset: " << start[n]*kernel_size*height*width << "label : " << start[n];
        int offsetIn = n*bottomOffset + start[n]*kernel_size_;
        int offsetOut = n*topOffset;
        for (int c = 0; c < kernel_size_; ++c) {
            bottom_diff[offsetIn+c] = top_diff[offsetOut+c];
        }
    }
  }
}


INSTANTIATE_CLASS(MaskOutputsLayer);
REGISTER_LAYER_CLASS(MaskOutputs);

}