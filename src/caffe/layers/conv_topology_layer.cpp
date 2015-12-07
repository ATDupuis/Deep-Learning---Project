#include <vector>

#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionTopologyLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  this->output_shape_.clear();
    
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_shape_data[i])
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionTopologyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
       
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, topology_filter_.cpu_data(), bottom_data);
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight, top_data + n * this->top_dim_);
        
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionTopologyLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          
          ConstructWeightMask();
          this->weight_cpu_gemm(bottom_data, topology_filter_.cpu_data(), bottom_data);
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_, top_diff + n * this->top_dim_, weight_diff  );
            
          
          //this->weight_cpu_gemm(weight_diff, topology_filter_.cpu_data() ,weight_diff );
            
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}
    
template <typename Dtype>
    void ConvolutionTopologyLayer<Dtype>::ConstructWeightMask()
    {
        vector<int> weight_shape(2);
        
        const int N_ = *(this->kernel_shape_.cpu_data());
        
        weight_shape[0] = N_;
        weight_shape[1] = N_;
        topology_filter_.Reshape(weight_shape);
        
        Dtype* data = topology_filter_.mutable_cpu_data();
        
        for(int weight_index = 0; weight_index < N_; weight_index++) {
            data[weight_index * N_ + weight_index] = 1;
            if (weight_index - 1 >= 0)
                data[weight_index * N_ + weight_index - 1] = 0.5;
            if (weight_index - 2 >= 0)
                data[weight_index * N_ + weight_index - 2] = 0.25;
            if (weight_index + 1 < N_)
                data[weight_index * N_ + weight_index + 1] = 0.5;
            if (weight_index + 2 < N_)
                data[weight_index* N_ + weight_index + 2] = 0.25;
        }
        
    }

// We do not have a GPU version yet, so turn this off.
//#ifdef CPU_ONLY
//STUB_GPU(ConvolutionTopologyLayer);
//#endif

INSTANTIATE_CLASS(ConvolutionTopologyLayer);

}  // namespace caffe
