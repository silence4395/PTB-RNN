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
    float min_data = 99999999;
    float max_data = -99999999;
    enum op_types {
      TANH = 0,
      PLAN = 1,
      EXPONENT = 2,
      AREAS = 3,
      PLAN_LUT = 4,
      LUT_BIT_LEVEL = 5
    }op_type;

    op_type = LUT_BIT_LEVEL;

    switch (op_type) {
    case TANH:
      {
        for (int i = 0; i < N; i++)
          {
            output(i) = (exp(input(i)) - exp(-input(i)))/(exp(input(i)) + exp(-input(i)));

            //if (input(i) > max_data)
            //  max_data = input(i);
            //if (input(i) < min_data)
            //  min_data = input(i);
          }
        //LOG(INFO) << "[ Info] max input: " << max_data << ", min input: " << min_data;
        break;
      }
    case PLAN:   // 79.200
      {
	float variable[3]    = {0.5  , 1.1875 , 2.5};
	float coefficient[3] = {0.25, 0.125, 0.03125};
	float const_b[3]     = {0.5 , 0.625, 0.84375};
	
	int length = sizeof(coefficient) / sizeof(coefficient[0]);
	
	for (int i = 0; i < N; i++) {
	  if (fabs(input(i)) >= 2.5)
	    tmp_data = 1;
	  else {
	    for (int j = 0; j < length; ++j)
	      if (fabs(input(i)) < variable[j]) {
		tmp_data = coefficient[j] * fabs(input(i) * 2) + const_b[j];
		break;
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
    case EXPONENT:   // 81.400
      {
        for (int i = 0; i < N; i++) {
          output(i) = 1 - 2/(1 + pow(2, 3 * input(i)));
        }
        break;
      }
    case AREAS: // 78.913
      {
	float variable[10]    = {0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5};
	float coefficient[10] = {0.92426, 0.58536, 0.27596, 0.11226, 0.04292,
				 0.01602, 0.00592, 0.00218, 0.00080, 0.00030};
	float const_b[10]     = {0.00916, 0.18831, 0.49836, 0.74267, 0.88056,
				 0.94747, 0.97762, 0.99066, 0.99616, 0.99844};
	
	int length = sizeof(coefficient) / sizeof(coefficient[0]);
	
	for (int i = 0; i < N; i++) {
	  if (fabs(input(i)) > 5)
	    tmp_data = 1;
	  else {
	    for (int j = 0; j < length; ++j) {
	      if (fabs(input(i)) <= variable[j]) {
		tmp_data = coefficient[j] * fabs(input(i)) + const_b[j];
		break;
	      }
	    } 
	  }
	  
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
        float lut_data[64];
        float base_addr = (4 - 1.1875) / LUT_NO;

        // calculate margin between PLAN and tanh
        for (int m = 0; m < LUT_NO; ++m) {
          float margin_addr = 1.1875 + base_addr * m;
          float tanh_data = (exp(margin_addr) - exp(-margin_addr))/(exp(margin_addr) + exp(-margin_addr));
          float piecewise_data = 0.03125 * margin_addr * 2 + 0.84375;
          piecewise_data = 2 * piecewise_data - 1;

          if (margin_addr < 2.5)
            lut_data[m] = tanh_data - piecewise_data;
          else
            lut_data[m] = tanh_data - 1;
        }

        // PLAN
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

            // LUT
            if (fabs(input(i)) >= 1.1875 && fabs(input(i)) < 4) {
              float addr_mode = fmod(float(fabs(input(i)) - 1.1875), base_addr);
              float tmp_addr = float(float(float(fabs(input(i)) - 1.1875)) / base_addr);

              if (addr_mode == 0.0)
                tmp_data = tmp_data + lut_data[int(tmp_addr)];
              else {
                tmp_addr = ceil(tmp_addr);
                if (int(tmp_addr) >= LUT_NO)
                  tmp_addr = LUT_NO - 1;
                tmp_data = tmp_data + (lut_data[int(tmp_addr)] + lut_data[int(tmp_addr) - 1]) / 100;
              }
            }

            // final
            if (input(i) >= 0)
              output(i) = tmp_data;
            else
              output(i) = -tmp_data;

            if (tmp_data > 1)
              LOG(FATAL) << " [ Error ] Exceed boundary.";
          }
        break;
      }
    case LUT_BIT_LEVEL:
      {
	float variable[21] = {0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375,
			      1  , 1.125 , 1.25 , 1.375 , 1.5 , 1.625 , 1.75 , 1.875 ,
			      2  , 3     , 4    , 5     , 6   , 8 };
	float avg_value[21] = {0.5    , 0.53234, 0.57561, 0.61589, 0.65318 , 0.68756, 0.71910, 0.74794,
			       0.78603, 0.82930, 0.86448, 0.89284, 0.91554 , 0.93360, 0.94790, 0.95919,
			       0.97954, 0.99719, 0.99962, 0.99997, 0.999995 };
	int length = sizeof(avg_value) / sizeof(avg_value[0]);

	for (int i = 0; i < N; ++i) {
	  if (fabs(input(i)) >= 0 && fabs(input(i)) < 0.5)
	    tmp_data = fabs(input(i));
	  else if (fabs(input(i)) >= 8)
	    tmp_data = 1;
	  else {
	    for (int j = 0; j < length; ++j) {
	      if (fabs(input(i)) < variable[j]) {
		tmp_data = avg_value[j];
		break;
	      }
	    }
	  }
	  
	  if (input(i) > 0)
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
