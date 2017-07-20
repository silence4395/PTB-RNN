/*
 *        (C) COPYRIGHT Ingenic Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : sigmoid_diy.cpp
 * Authors    : zhluo@aries
 * Create Time: 2017-07-20:21:36:50
 * Description:
 * 
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

REGISTER_OP("SigmoidDiy")
        .Input("to_sigmoid : float32")
        .Output("out_sigmoid : float32");

using namespace tensorflow;

class SigmoidDiyOp : public OpKernel {
public:
  explicit SigmoidDiyOp(OpKernelConstruction * context) : OpKernel(context) {}

  void Compute(OpKernelContext * context) override {
    const Tensor & input_tensor = context->input(0);
    auto input = input_tensor.flat<float32>();
    
    Tensor * output_tensor = NULL;
    OP_REQUIRES_OK(context, context_>allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output = output_tensor->flat<float32>();
    
    const int N = input.size();
    for () {}
  }
};

REGISTER_KERNEL_BUILDER(Name("SigmoidDiy").Device(DEVICE_CPU), SigmoidDiyOp);
