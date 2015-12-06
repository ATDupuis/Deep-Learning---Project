#include <vector>
#include <fstream>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void TopologyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {

    const int num_output = this->layer_param_.topology_param().num_output();
    bias_term_ = this->layer_param_.topology_param().bias_term();
    N_ = num_output;

    const int axis = bottom[0]->CanonicalAxisIndex(this->layer_param_.topology_param().axis());

    // Dimensions starting from "axis" are "flattened" into a single
    // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
    // and axis == 1, N inner products with dimension CHW are performed.
    K_ = bottom[0]->count(axis);

    // Check if we need to set up the weights
    if (this->blobs_.size() > 0) {
        LOG(INFO) << "Skipping parameter initialization";

    } else {
        if (bias_term_) {
            this->blobs_.resize(2);
        } else {
            this->blobs_.resize(1);
        }

        // Intialize the weight
        vector<int> weight_shape(2);
        weight_shape[0] = N_;
        weight_shape[1] = K_;
        this->blobs_[0].reset(new Blob<Dtype>(weight_shape));

        // Fill the weights
        shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(this->layer_param_.topology_param().weight_filler()));
        weight_filler->Fill(this->blobs_[0].get());

        // If necessary, intiialize and fill the bias term
        if (bias_term_) {
            vector<int> bias_shape(1, N_);
            this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
            shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(this->layer_param_.topology_param().bias_filler()));
            bias_filler->Fill(this->blobs_[1].get());
        }
    }

    this->param_propagate_down_.resize(this->blobs_.size(), true);

    ConstructWeightMask();
}

template <typename Dtype>
void TopologyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
    // Figure out the dimensions
    const int axis = bottom[0]->CanonicalAxisIndex(
                this->layer_param_.topology_param().axis());
    const int new_K = bottom[0]->count(axis);
    CHECK_EQ(K_, new_K)
            << "Input size incompatible with inner product parameters.";
    // The first "axis" dimensions are independent inner products; the total
    // number of these is M_, the product over these dimensions.
    M_ = bottom[0]->count(0, axis);
    // The top shape will be the bottom shape with the flattened axes dropped,
    // and replaced by a single axis with dimension num_output (N_).
    vector<int> top_shape = bottom[0]->shape();
    top_shape.resize(axis + 1);
    top_shape[axis] = N_;
    top[0]->Reshape(top_shape);
    // Set up the bias multiplier
    if (bias_term_) {
        vector<int> bias_shape(1, M_);
        bias_multiplier_.Reshape(bias_shape);
        caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
    }

    weighted_top_diff_.Reshape(top[0]->shape());
}

template <typename Dtype>
void TopologyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const Dtype* weight = this->blobs_[0]->cpu_data();
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
                          bottom_data, weight, (Dtype)0., top_data);
    if (bias_term_) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                              bias_multiplier_.cpu_data(),
                              this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
    }
}

template <typename Dtype>
void TopologyLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                            const vector<bool>& propagate_down,
                                            const vector<Blob<Dtype>*>& bottom) {
    if (this->param_propagate_down_[0]) {
        const Dtype* top_diff = top[0]->cpu_diff();
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* weighted_top_diff_data = weighted_top_diff_.mutable_cpu_data();
        // Gradient with respect to weight

//        caffe_cpu_axpby<Dtype>(N_, (Dtype)1., topology_weight_mask, (Dtype)1., bottom_data);
        //caffe_mul(N_, weight_mask_.cpu_data(), bottom_data, weighted_bottom_data);
        caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, M_, N_, N_, (Dtype)1.0, top_diff, weight_mask_.cpu_data(), (Dtype)0.0, weighted_top_diff_data);

        /*
template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}
        */

        /*std::cout << "Shape bottom: ";
        for (int axis_index = 0; axis_index < bottom[0]->num_axes(); ++axis_index)
        {
            std::cout << bottom[0]->shape(axis_index) << " ";
        }
        std::cout << std::endl;

        std::cout << "Shape top diff: ";
        for (int axis_index = 0; axis_index < top[0]->num_axes(); ++axis_index)
        {
            std::cout << top[0]->shape(axis_index) << " ";
        }
        std::cout << std::endl;

        std::cout << "Shape weights mask: ";
        for (int axis_index = 0; axis_index < weight_mask_.num_axes(); ++axis_index)
        {
            std::cout << weight_mask_.shape(axis_index) << " ";
        }
        std::cout << std::endl;

        std::cout << "Shape weights: ";
        for (int axis_index = 0; axis_index < this->blobs_[0]->num_axes(); ++axis_index)
        {
            std::cout << this->blobs_[0]->shape(axis_index) << " ";
        }
        std::cout << std::endl;

        std::ofstream log_pre("/home/allard/LogPre.txt", std::ifstream::trunc);
        std::ofstream log_post("/home/allard/LogPost.txt", std::ifstream::trunc);
        for (int row = 0; row < M_; ++row)
        {
            for (int col = 0; col < N_; ++col)
            {
                log_pre << top_diff[row * N_ + col] << " ";
                log_post << weighted_top_diff_data[row * N_ + col] << " ";
            }
            log_pre << "\n";
            log_post << "\n";
        }
        log_pre << std::flush;
        log_post << std::flush;
        exit(0);*/

        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
                              weighted_top_diff_data, bottom_data, (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }

    if (bias_term_ && this->param_propagate_down_[1]) {
        const Dtype* top_diff = top[0]->cpu_diff();
        // Gradient with respect to bias

        caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
                              bias_multiplier_.cpu_data(), (Dtype)1.,
                              this->blobs_[1]->mutable_cpu_diff());
    }
    if (propagate_down[0]) {
        const Dtype* top_diff = top[0]->cpu_diff();
        // Gradient with respect to bottom data
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
                              top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,
                bottom[0]->mutable_cpu_diff());
    }
}

template <typename Dtype>
void TopologyLayer<Dtype>::ConstructWeightMask() {
    vector<int> weight_mask_shape(2);

    weight_mask_shape[0] = N_;
    weight_mask_shape[1] = N_;
    weight_mask_.Reshape(weight_mask_shape);

    Dtype* data = weight_mask_.mutable_cpu_data();

    std::string mask_filler_type = this->layer_param_.topology_param().mask_filler().type();
    if (mask_filler_type == "gaussian")
    {
        Dtype standard_dev = this->layer_param_.topology_param().mask_filler().spread();

        for(int weight_index = 0; weight_index < N_; weight_index++) {
            for (int weight_index_offset = -weight_index; weight_index_offset < N_ - weight_index; ++weight_index_offset) {
                data[weight_index * N_ + weight_index + weight_index_offset] =
                        std::exp(-static_cast<Dtype>(weight_index_offset) * weight_index_offset / (2 * standard_dev * standard_dev));
            }
        }
    }
    else if (mask_filler_type == "triangular")
    {
        Dtype half_base = this->layer_param_.topology_param().mask_filler().spread();

        for(int weight_index = 0; weight_index < N_; weight_index++) {
            for (int weight_index_offset = -weight_index; weight_index_offset < N_ - weight_index; ++weight_index_offset) {
                data[weight_index * N_ + weight_index + weight_index_offset] =
                        static_cast<Dtype>(std::max(1.0 - 1.0 / half_base * std::abs(weight_index_offset), 0.0));
            }
        }
    }
    else
    {
        CHECK(false) << "Unknown mask filler type: " << mask_filler_type;
    }

    std::ofstream weight_mask_matrix("/home/allard/LogWMM.txt", std::ofstream::trunc);
    for (int row = 0; row < N_; ++row)
    {
        for (int col = 0; col < N_; ++col)
        {
            weight_mask_matrix << data[row * N_ + col] << " ";
        }
        weight_mask_matrix << "\n";
    }
    weight_mask_matrix << std::flush;

}

#ifdef CPU_ONLY
STUB_GPU(TopologyLayer);
#endif

INSTANTIATE_CLASS(TopologyLayer);
REGISTER_LAYER_CLASS(Topology);

}  // namespace caffe
