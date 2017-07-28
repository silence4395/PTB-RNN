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

REGISTER_OP("TanhDiy")
      .Input("in : float")
      .Output("out : float")
      .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
             c->set_output(0, c->input(0));
	     return Status::OK();
  });

class TanhDiyOp : public OpKernel {
public:
  explicit TanhDiyOp(OpKernelConstruction * context) : OpKernel(context){}

  void Compute(OpKernelContext * context) override {
    const Tensor & input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();
    
    Tensor * output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output = output_tensor->flat<float>();
    
    const int N = input.size();
    float tmp_data = 0;
    enum op_types {
      TANH = 0,
      PLAN = 1,
      EXPONENT = 2,
      AREAS = 3,
      PLAN_LUT = 4
    }op_type;
    
    op_type = PLAN_LUT;
    
    switch (op_type) {
    case TANH:
      {
	for (int i = 0; i < N; i++)
	  {
	    output(i) = (exp(input(i)) - exp(-input(i)))/(exp(input(i)) + exp(-input(i)));
	  }
	break;
      }
    case PLAN:   // 79.200
      {
	for (int i = 0; i < N; i++)
	  {
	    if (fabs(input(i)) >= 0 && fabs(input(i)) <0.5)
	      tmp_data = 0.25*fabs(input(i)*2) + 0.5;
	    else if (fabs(input(i)) >= 0.5 && fabs(input(i)) < 1.1875)
	      tmp_data = 0.125*fabs(input(i)*2) + 0.625;
	    else if (fabs(input(i)) >= 1.1875 && fabs(input(i)) < 2.5)
	      tmp_data = 0.03125*fabs(input(i)*2) + 0.84375;
	    else
	      tmp_data = 1;
	      
	    tmp_data = 2 * tmp_data - 1;
	    if (input(i) >= 0)
	      output(i) = tmp_data;
	    else
	      output(i) = -tmp_data;
	  }
	break;
      }
    case EXPONENT:   // 81.400
      {
	for (int i = 0; i < N; i++) {
	  output(i) = 1 - 2/(1 + pow(2, 3 * input(i)));
	}
	break;
      }
    case AREAS: // 78.913
      {
	for (int i = 0; i < N; i++) {
	  if (fabs(input(i)) >= 0 && fabs(input(i)) <= 0.5)
	    tmp_data = 0.92426 * fabs(input(i)) + 0.00916;
	  else if (fabs(input(i)) > 0.5 && fabs(input(i)) <= 1)
	    tmp_data = 0.58536 * fabs(input(i)) + 0.18831;
	  else if (fabs(input(i)) > 1 && fabs(input(i)) <= 1.5)
	    tmp_data = 0.27596 * fabs(input(i)) + 0.49836;
	  else if (fabs(input(i)) > 1.5 && fabs(input(i)) <= 2)
	    tmp_data = 0.11226 * fabs(input(i)) + 0.74267;
	  else if (fabs(input(i)) > 2 && fabs(input(i)) <= 2.5)
	    tmp_data = 0.04292 * fabs(input(i)) + 0.88056;
	  else if (fabs(input(i)) > 2.5 && fabs(input(i)) <= 3)
	    tmp_data = 0.01602 * fabs(input(i)) + 0.94747;
	  else if (fabs(input(i)) > 3 && fabs(input(i)) <= 3.5)
	    tmp_data = 0.00592 * fabs(input(i)) + 0.97762;
	  else if (fabs(input(i)) > 3.5 && fabs(input(i)) <= 4)
	    tmp_data = 0.00218 * fabs(input(i)) + 0.99066;
	  else if (fabs(input(i)) > 4 && fabs(input(i)) <= 4.5)
	    tmp_data = 0.00080 * fabs(input(i)) + 0.99616;
	  else if (fabs(input(i)) > 4.5 && fabs(input(i)) <= 5)
	    tmp_data = 0.00030 * fabs(input(i)) + 0.99844;
	  else if (fabs(input(i)) > 5)
	    tmp_data = 1;
	  
	  if (input(i) == 0)
	    output(i) = 0;
	  else if (input(i) > 0)
	    output(i) = tmp_data;
	  else
	    output(i) =  -tmp_data;
	}
	break;
      }
    case PLAN_LUT: // 77.929
      {
	const int LUT_NO = 64;
	double lut_data[64];
	float base_addr = 0;
	
	base_addr = (4 - 1.1875) / LUT_NO;
	// statistical error between PLAN and sigmoid(2.375~5, 5~10 or 5~20)
	for (int m = 0; m < LUT_NO; ++m) {
	  if ((1.1875 + base_addr * m) < 2.5)
	    lut_data[m] = 2/(1 + exp(-2 * (1.1875 + base_addr * m))) - 1 -
	      (0.03125*(1.1875 + base_addr * m)*2 + 0.84375);
	  else
	    lut_data[m] = 2/(1 + exp(-2 * (1.1875 + base_addr * m))) - 1 - 1;
	}
	
	// >>>>>>>
	for (int i = 0; i < N; i++)
	  {
	    // PLAN
	    if (fabs(input(i)) >= 0 && fabs(input(i)) <0.5)
	      tmp_data = 0.25*fabs(input(i)*2) + 0.5;
	    else if (fabs(input(i)) >= 0.5 && fabs(input(i)) < 1.1875)
	      tmp_data = 0.125*fabs(input(i)*2) + 0.625;
	    else if (fabs(input(i)) >= 1.1875 && fabs(input(i)) < 2.5)
	      tmp_data = 0.03125*fabs(input(i)*2) + 0.84375;
	    else
	      tmp_data = 1;
	    
	    // LUT
	    if (fabs(input(i)) >= 1.1875 && fabs(input(i)) < 4) {
	      float addr_mod = fmod(float(fabs(input(i)) - 1.1875), base_addr);
	      int tmp_addr = float(fabs(input(i)) - 1.1875) / base_addr;
	      if (addr_mod == 0)
		tmp_data = tmp_data + lut_data[tmp_addr];
	      else {
		tmp_addr = ceil(tmp_addr);
		tmp_data = tmp_data + (lut_data[tmp_addr] + lut_data[tmp_addr-1])/2;
	      }
	    }
	    
	    tmp_data = 2 * tmp_data - 1;
	    if (input(i) >= 0)
	      output(i) = tmp_data;
	    else
	      output(i) = -tmp_data;
	  }
	break;
      }
    default:
      LOG(FATAL) << "Please set function type.";
    } // switch
  }
};

REGISTER_KERNEL_BUILDER(Name("TanhDiy").Device(DEVICE_CPU), TanhDiyOp);

