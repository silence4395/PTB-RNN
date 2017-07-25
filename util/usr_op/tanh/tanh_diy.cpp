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
      AREAS = 3
    }op_type;
    
    op_type = AREAS;
    
    switch (op_type) {
    case TANH:
      {
	for (int i = 0; i < N; i++)
	  {
	    output(i) = (exp(input(i)) - exp(-input(i)))/(exp(input(i)) + exp(-input(i)));
	  }
	break;
      }
    case PLAN:   // 102.282
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
	      
	    if (input(i) >= 0)
	      output(i) = 2 * tmp_data - 1;
	    else
	      {
		if (tmp_data == 1)
		  output(i) = 0;
		else
		  output(i) = 1 - 2 * tmp_data;
	      }
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
    case AREAS: // 89.187
      {
	for (int i = 0; i < N; i++) {
	  if (fabs(input(i)) > 0 && fabs(input(i)) <= 0.5)
	    tmp_data = 0.22830 * fabs(input(i)) * 2 + 0.50596;
	  else if (fabs(input(i)) > 0.5 && fabs(input(i)) <= 1)
	    tmp_data = 0.14945 * fabs(input(i)) * 2 + 0.58948;
	  else if (fabs(input(i)) > 1 && fabs(input(i)) <= 1.5)
	    tmp_data = 0.06749 * fabs(input(i)) * 2 + 0.75292;
	  else if (fabs(input(i)) > 1.5 && fabs(input(i)) <= 2)
	    tmp_data = 0.02739 * fabs(input(i)) * 2 + 0.87367;
	  else if (fabs(input(i)) > 2 && fabs(input(i)) <= 2.5)
	    tmp_data = 0.01046 * fabs(input(i)) * 2 + 0.94147;
	  else if (fabs(input(i)) > 2.5 && fabs(input(i)) <= 3)
	    tmp_data = 0.00390 * fabs(input(i)) * 2 + 0.97428;
	  else if (fabs(input(i)) > 3 && fabs(input(i)) <= 3.5)
	    tmp_data = 0.00144 * fabs(input(i)) * 2 + 0.98905;
	  else if (fabs(input(i)) > 3.5 && fabs(input(i)) <= 4)
	    tmp_data = 0.00053 * fabs(input(i)) * 2 + 0.99543;
	  else if (fabs(input(i)) > 4 )
	    tmp_data = 1;
	  
	  if (input(i) >= 0)
	    output(i) = 2 * tmp_data - 1;
	  else
	    {
	      if (tmp_data == 1)
		output(i) = 0;
	      else
		output(i) = 1 - 2 * tmp_data;
	    }
	}
	break;
      }
    default:
      LOG(FATAL) << "Please set function type.";
    } // switch
  }
};

REGISTER_KERNEL_BUILDER(Name("TanhDiy").Device(DEVICE_CPU), TanhDiyOp);

