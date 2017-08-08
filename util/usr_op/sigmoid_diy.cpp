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

    const int N = input.size();
    float max_value = -9999999999;
    float min_value = 999999999;
    float tmp_data = 0.0;

    enum op_types {
      SIGMOID = 0,
      PLAN = 1,
      SONF = 2,
      INTERPOLATION = 3,
      EXPONENT = 4,
      AREAS = 5,
      PLAN_LUT = 6,
      LUT_BIT_LEVEL = 7
    } op_type;

    // set function type,
    op_type = LUT_BIT_LEVEL;

    switch(op_type){
    case SIGMOID:    // 78.793
      {
	//LOG(INFO) << "[ Info ] Current select < SIGMOID > function";
	for (int i = 0; i < N; i++) {
	  output(i) = 1/(1+exp(-input(i)));

	  //if (max_value < input(i))
	  //  max_value = input(i);
	  //if (min_value > input(i))
	  //  min_value = input(i);
	}
	break;
      }
    case PLAN: // 79.470
      {
	float variable[3]    = {1  , 2.375 , 5};
	float coefficient[3] = {0.25, 0.125, 0.03125};
	float const_b[3]     = {0.5 , 0.625, 0.84375};

	int length = sizeof(coefficient) / sizeof(coefficient[0]);

	for (int i = 0; i < N; i++) {
	  if (fabs(input(i)) >= 5)
	    tmp_data = 1;
	  else {
	    for (int j = 0; j < length; ++j)
	      if (fabs(input(i)) < variable[j]) {
		tmp_data = coefficient[j] * fabs(input(i)) + const_b[j];
		break;
	      }
	  }

	  if (input(i) >= 0)
	    output(i) = tmp_data;
	  else
	    output(i) = 1 - tmp_data;
	}
	break;
      }
    case SONF: // 79.524
      {
	//LOG(INFO) << "[ Info ] Current select < SONF > function";
	for (int i = 0; i < N; i++) {
	  if (input(i) > 0 && input(i) <= 4)
	    output(i) = 1 - 0.5 * (1 - 0.25 * fabs(input(i))) * (1 - 0.25 * fabs(input(i)));
	  else if (input(i) > -4 && input(i) <= 0)
	    output(i) = 0.5 * (1 - 0.25 * fabs(input(i))) * (1 - 0.25 * fabs(input(i)));
	  else if (input(i) > 4)
	    output(i) = 1;
	  else
	    output(i) = 0;
	}
	break;
      }
    case INTERPOLATION:
      {
	//LOG(INFO) << "[ Info ] Current select < INTERPOLATION > function";
	float x[8], y[8];
	for (int j = 0; j < 8; ++j)
	  {
	    y[j] = 0.0625 *(j + 8);
	    x[j] = log(y[j]/(1-y[j]));
	    //LOG(INFO) << "x: " << x[j] << ", y: " << y[j];
	  }

	for (int i = 0; i < N; i++){
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

	  else if (fabs(input(i)) >= x[7] && fabs(input(i)) < 3.5)    // 79.005
	    tmp_data = (fabs(input(i)) - x[7]) * 0.041906405 + 0.9375;
	  else if (fabs(input(i)) >= 3.5 && fabs(input(i)) < 4.5)
	    tmp_data = (fabs(input(i)) - 3.5) * 0.018325288 + 0.970687769;
	  else if (fabs(input(i)) >= 4.5 && fabs(input(i)) < 6)
	    tmp_data = (fabs(input(i)) - 4.5) * 0.00851432 + 0.989013057;
	  else if (fabs(input(i)) >= 6 && fabs(input(i)) < 8)
	    tmp_data = (fabs(input(i)) - 6) * 0.001068636 + 0.997527377;
	  else
	    tmp_data = 1;
	  //else if (fabs(input(i)) >= x[7] && fabs(input(i)) < 4)     // 79.157
	  //  tmp_data = (fabs(input(i)) - x[7]) * 0.034241377 + 0.9375;
	  //else if (fabs(input(i)) >= 4 && fabs(input(i)) < 6)
	  //  tmp_data = (fabs(input(i)) - 4) * 0.007756793 + 0.98201379;
	  //else if (fabs(input(i)) >= 6 && fabs(input(i)) < 8)
	  //  tmp_data = (fabs(input(i)) - 6) * 0.001068636 + 0.997527377;
	  //else if (fabs(input(i)) >= 8)
	  //  tmp_data = 1;

	  //else  // 89.xxx
	  //  tmp_data = (fabs(input(i))-x[7])*(1-y[7])/(7-x[7])+y[7];

	  if (input(i) >= 0)
	    output(i) = tmp_data;
	  else
	    output(i) = 1 - tmp_data;
	}
	break;
      }
    case EXPONENT: // 78.793
      {
	for (int i = 0; i < N; i++) {
	  output(i) = 1/(1 + pow(2, -1.5 * input(i)));
	}
	break;
      }
    case AREAS: // 78.793
      {
	float variable[8]    = {1, 2, 3, 4, 5, 6, 7, 8};
	float coefficient[8] = {0.22830, 0.14945, 0.06749, 0.02739, 0.01046, 0.00390, 0.00144, 0.00053};
	float const_b[8]     = {0.50596, 0.58948, 0.75292, 0.87367, 0.94147, 0.97428, 0.98905, 0.99543};

	int length = sizeof(coefficient) / sizeof(coefficient[0]);

	for (int i = 0; i < N; i++) {
	  if (fabs(input(i)) > 8)
	    tmp_data = 1;
	  else {
	    for (int j = 0; j < length; ++j) {
	      if (fabs(input(i)) <= variable[j]) {
		tmp_data = coefficient[j] * fabs(input(i)) + const_b[j];
		break;
	      }
	    }
	  }

	  if (input(i) >= 0)
	    output(i) = tmp_data;
	  else
	    output(i) = 1 - tmp_data;
	}
	break;
      }
    case PLAN_LUT: // 78.754
      {
	const int LUT_NO = 64;
	float lut_data[64];
	float base_addr = 0;

	base_addr = (8 - 2.375) / LUT_NO;
	// statistical error between PLAN and sigmoid(2.375~5, 5~10 or 5~20)
	for (int m = 0; m < LUT_NO; ++m) {
	  if ((2.375 + base_addr * m) < 5)
	    lut_data[m] = 1/(1+exp(-(2.375 + base_addr * m))) -
	      (0.03125*(2.375 + base_addr * m) + 0.84375);
	  else
	    lut_data[m] = 1/(1+exp(-(2.375 + base_addr * m))) - 1;
	  //LOG(INFO) << " [ Info ] lut addr: " << 2.375 + base_addr * m << ", lut_data: " << lut_data[m];
	}

	// >>>>>
	for (int i = 0; i < N; i++){
	  // PLAN
	  if (fabs(input(i)) >= 0 && fabs(input(i)) <1)
	    tmp_data = 0.25*fabs(input(i)) + 0.5;
	  else if (fabs(input(i)) >= 1 && fabs(input(i)) < 2.375)
	    tmp_data = 0.125*fabs(input(i)) + 0.625;
	  else if (fabs(input(i)) >= 2.375 && fabs(input(i)) < 5)
	    tmp_data = 0.03125*fabs(input(i)) + 0.84375;
	  else
	    tmp_data = 1;

	  // LUT
	  if (fabs(input(i)) >= 2.375 && fabs(input(i)) < 8) {
	    float addr_mod = fmod(float(fabs(input(i)) - 2.375), base_addr);
	    float tmp_addr = float(float(fabs(input(i)) - 2.375) / base_addr);
	    if (addr_mod == 0.0)
	      tmp_data = tmp_data + lut_data[int(tmp_addr)];
	    else {
	      tmp_addr = ceil(tmp_addr);
	      if (int(tmp_addr) >= LUT_NO)
		tmp_addr = LUT_NO - 1;
	      tmp_data = tmp_data + (lut_data[int(tmp_addr)] + lut_data[int(tmp_addr)-1])/2;
	    }
	  }

	   //LOG(INFO) << " [ Info ] sigmoid data: " << tmp_data;

	  // final result
	  if (input(i) >= 0)
	    output(i) = tmp_data;
	  else
	    output(i) = 1 - tmp_data;
	}

	break;
      }
    case LUT_BIT_LEVEL:
      {
	// sigmoid(x) = (tanh(x / 2) + 1) / 2
	float variable[21] = {0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375,
			      1  , 1.125 , 1.25 , 1.375 , 1.5 , 1.625 , 1.75 , 1.875 ,
			      2  , 3     , 4    , 5     , 6   , 8 };
	float avg_value[21] = {0.5    , 0.53234, 0.57561, 0.61589, 0.65318 , 0.68756, 0.71910, 0.74794,
			       0.78603, 0.82930, 0.86448, 0.89284, 0.91554 , 0.93360, 0.94790, 0.95919,
			       0.97954, 0.99719, 0.99962, 0.99997, 0.999995 };

	int length = sizeof(avg_value) / sizeof(avg_value[0]);

	float tanh_input = 0.0;
	for (int i = 0; i < N; ++i) {
	  tanh_input = fabs(input(i)) / 2;

	  if (tanh_input >= 0 && tanh_input < 0.5)
	    tmp_data = tanh_input;
	  else if (tanh_input >= 8)
	    tmp_data = 1;
	  else {
	    for (int j = 0; j < length; ++j) {
	      if (tanh_input < variable[j]) {
		tmp_data = avg_value[j];
		break;
	      }
	    }
	  }

	  tmp_data = (tmp_data + 1) / 2;

	  if (input(i) > 0)
	    output(i) = tmp_data;
	  else
	    output(i) = 1 - tmp_data;
	}
	break;
      }
    default:
      LOG(FATAL) << " [ Error ] Please set function type.";
    } //switch
    //LOG(INFO) << " [Info] max: " << max_value << ", min: " << min_value;
  }
};

REGISTER_KERNEL_BUILDER(Name("SigmoidDiy").Device(DEVICE_CPU), SigmoidDiyOp);
