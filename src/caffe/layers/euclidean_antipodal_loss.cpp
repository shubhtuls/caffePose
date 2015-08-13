#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/pose_layers.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanAntipodalLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  diff1_.ReshapeLike(*bottom[0]);
  diffPos_.ReshapeLike(*bottom[0]);
  diffNeg_.ReshapeLike(*bottom[0]);
  N_ = bottom[0]->shape()[0];
}

template <typename Dtype>
void EuclideanAntipodalLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int dim_ = count/N_;
  
  Dtype* diffPosData = diffPos_.mutable_cpu_data();
  Dtype* diffNegData = diffNeg_.mutable_cpu_data();
  Dtype* diff0Data = diff_.mutable_cpu_data();
  Dtype* diff1Data = diff1_.mutable_cpu_data();
  Dtype dotTotal = 0;

  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diffPosData);

  caffe_add(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diffNegData); // subtracting antipodal points of bottom[1]

  for(int n=0;n<N_;n++){
    Dtype dotPos=0,dotNeg=0;
    for(int j = 0;j<dim_;j++){
      dotPos += diffPosData[n*dim_+j]*diffPosData[n*dim_+j];
      dotNeg += diffNegData[n*dim_+j]*diffNegData[n*dim_+j];
    }
    if(dotNeg > dotPos){
      dotTotal += dotPos;
      for(int j = 0;j<dim_;j++){
        diff0Data[n*dim_+j] = diffPosData[n*dim_+j];
        diff1Data[n*dim_+j] = -diffPosData[n*dim_+j];
      }
    }
    else{
      dotTotal += dotNeg;
      for(int j = 0;j<dim_;j++){
        diff0Data[n*dim_+j] = diffNegData[n*dim_+j];
        diff1Data[n*dim_+j] = diffNegData[n*dim_+j];
      }
    }
  }

  Dtype loss = dotTotal / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanAntipodalLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype alpha = top[0]->cpu_diff()[0] / bottom[i]->num();
      const Dtype* diff_data = (i == 0) ? diff_.cpu_data() : diff1_.cpu_data();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_data,                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanAntipodalLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanAntipodalLossLayer);
REGISTER_LAYER_CLASS(EuclideanAntipodalLoss);

}  // namespace caffe
