#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/pose_layers.hpp"
#include <math.h>

namespace caffe {

namespace QuatNormalization {
	typedef std::vector<float> Quaternion;

	float DotProduct(Quaternion q1, Quaternion q2){
		float val = 0;
		for(int j = 0;j<q1.size();j++){
			val += q1[j]*q2[j];
		}
		return val;
	}

	Quaternion VecMultiply(std::vector<Quaternion> mat, Quaternion q){
		Quaternion result;
		for(int i = 0;i<mat.size();i++){
			result.push_back(DotProduct(q, mat[i]));
		}
		return result;
	}

  std::vector<Quaternion> RankOneMat(Quaternion q){
    std::vector<Quaternion> result;
    for(int i =0;i<q.size();i++){
      Quaternion p;
      for(int j = 0;j<q.size();j++){
        p.push_back(q[i]*q[j]);
      }
    result.push_back(p);
    }
  return result;
  }

}

template <typename Dtype>
void QuatNormalizationLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> topShape(2);
  N_ = bottom[0]->shape()[0];
  qDim_ = 4;
  topShape[0] = N_;
  topShape[1] = qDim_;
  top[0]->Reshape(topShape);
}

template <typename Dtype>
void QuatNormalizationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* q_data = bottom[0]->cpu_data();
  Dtype* s_data = top[0]->mutable_cpu_data();

  for(int n = 0; n<N_; n++){
    QuatNormalization::Quaternion q,s;

    for(int j = 0;j<qDim_;j++){
      q.push_back(float(q_data[n*qDim_ + j]));
    }
    float qNorm = sqrt(QuatNormalization::DotProduct(q,q));
    for(int j = 0;j<qDim_;j++){
      s.push_back(float(q_data[n*qDim_ + j]/qNorm));
    }
    for(int j = 0;j<qDim_;j++){
      s_data[n*qDim_ + j] = Dtype(s[j]);
    }
  }
}

template <typename Dtype>
void QuatNormalizationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* q_data = bottom[0]->cpu_data();
  const Dtype* s_data = top[0]->cpu_data();

  Dtype* q_diff;
  if(propagate_down[0]){
  	q_diff = bottom[0]->mutable_cpu_diff();
  	caffe_set(bottom[0]->count(), Dtype(0), q_diff);
  }

  const Dtype* s_diff = top[0]->cpu_diff();

  for(int n = 0; n<N_; n++){
  	QuatNormalization::Quaternion q,s,dq,ds;
  	for(int j = 0;j<qDim_;j++){
      q.push_back(float(q_data[n*qDim_ + j]));
      s.push_back(float(s_data[n*qDim_ + j]));
      ds.push_back(float(s_diff[n*qDim_ + j]));
    }
    if(propagate_down[0]){
      std::vector<QuatNormalization::Quaternion> s_st = QuatNormalization::RankOneMat(s);
      float qNorm = sqrt(QuatNormalization::DotProduct(q,q));
    	dq = QuatNormalization::VecMultiply(s_st,ds);
    	for(int j = 0;j<qDim_;j++){
      		q_diff[n*qDim_ + j] = Dtype((ds[j] - dq[j])/qNorm);
    	}
    }

  }

}

INSTANTIATE_CLASS(QuatNormalizationLayer);
REGISTER_LAYER_CLASS(QuatNormalization);

}