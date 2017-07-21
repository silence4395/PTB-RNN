/*
 *        (C) COPYRIGHT Ingenic Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : sigmoid_diy.cpp
 * Authors    : zhluo@aries
 * Create Time: 2017-07-20:20:42:25
 * Description:
 * 
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("SigmoidDiy")
      .Input("in : float")
      .Output("out : float")
      .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
             c->set_output(0, c->input(0));
	     return Status::OK();
  });

class SigmoidDiyOp : public OpKernel {
public:
  explicit SigmoidDiyOp(OpKernelConstruction * context) : OpKernel(context){}

  void Compute(OpKernelContext * context) override {
    const Tensor & input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();
    
    Tensor * output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output = output_tensor->flat<float>();
    
    
    float x[8], y[8];
    for (int j = 0; j < 8; ++j)
      {
	y[j] = 0.0625 *(j + 8);
	x[j] = log(y[j]/(1-y[j]));
	//LOG(INFO) << "x: " << x[j] << ", y: " << y[j];
      }
    
    const int N = input.size();
    float max_value = -9999999999;
    float min_value = 999999999;
    float tmp_data = 0.0;
        
    for (int i = 0; i < N; i++)
      {
	/////////////////// [ 0 sigmoid function ] ///////////////////
	//output(i) = 1/(1+exp(-input(i)));
	
	//if (max_value < input(i))
	//  max_value = input(i);
	//if (min_value > input(i))
	//  min_value = input(i);
	
	/////////////////// [ 1 PLAN ] ///////////////////
	//if (fabs(input(i)) >= 0 && fabs(input(i)) <1)
	//  tmp_data = 0.25*fabs(input(i)) + 0.5;
	//else if (fabs(input(i)) >= 1 && fabs(input(i)) < 2.375)
	//  tmp_data = 0.125*fabs(input(i)) + 0.625;
	//else if (fabs(input(i)) >= 2.375 && fabs(input(i)) < 5)
	//  tmp_data = 0.03125*fabs(input(i)) + 0.84375;
	//else
	//  tmp_data = 1;
	//
	//if (input(i) >= 0)
	//  output(i) = tmp_data;
	//else
	//  output(i) = 1 - tmp_data;
	
	/////////////////// [ 2 SONF ] ///////////////////
	//if (input(i) > 0 && input(i) <= 4)
	//  output(i) = 1 - 0.5 * (1 - 0.25 * fabs(input(i))) * (1 - 0.25 * fabs(input(i)));
	//else if (input(i) > -4 && input(i) <= 0)
	//  output(i) = 0.5 * (1 - 0.25 * fabs(input(i))) * (1 - 0.25 * fabs(input(i))); 
	//else if (input(i) > 4)
	//  output(i) = 1;
	//else
	//  output(i) = 0;
	  
	/////////////////// [ 3 PLAN & Interpolation ] ///////////////////
	 
	if (fabs(input(i)) >= x[0] && fabs(input(i)) < x[1])
	  tmp_data = (fabs(input(i))-x[0])*(y[1]-y[0])/(x[1]-x[0])+y[0];
	else if (fabs(input(i)) >= x[1] && fabs(input(i)) < x[2])
	  tmp_data = (fabs(input(i))-x[1])*(y[2]-y[1])/(x[2]-x[1])+y[1];
	else if (fabs(input(i)) >= x[2] && fabs(input(i)) < x[3])
	  tmp_data = (fabs(input(i))-x[2])*(y[3]-y[2])/(x[3]-x[2])+y[2];
	else if (fabs(input(i)) >= x[3] && fabs(input(i)) < x[4])
	  tmp_data = (fabs(input(i))-x[3])*(y[4]-y[3])/(x[4]-x[3])+y[3];
	else if (fabs(input(i)) >= x[4] && fabs(input(i)) < x[5])
	  tmp_data = (fabs(input(i))-x[4])*(y[5]-y[4])/(x[5]-x[4])+y[4];
	else if (fabs(input(i)) >= x[5] && fabs(input(i)) < x[6])
	  tmp_data = (fabs(input(i))-x[5])*(y[6]-y[5])/(x[6]-x[5])+y[5];
	else if (fabs(input(i)) >= x[6] && fabs(input(i)) < x[7])
	  tmp_data = (fabs(input(i))-x[6])*(y[7]-y[6])/(x[7]-x[6])+y[6];
	else
	  tmp_data = (fabs(input(i))-x[7])*(1-y[7])/(20-x[7])+y[7];
	
	if (input(i) >= 0)
	  output(i) = tmp_data;
	else
	  output(i) = 1 - tmp_data;
      }
    //LOG(INFO) << " [Info] max: " << max_value << ", min: " << min_value;
  }
};

REGISTER_KERNEL_BUILDER(Name("SigmoidDiy").Device(DEVICE_CPU), SigmoidDiyOp);

