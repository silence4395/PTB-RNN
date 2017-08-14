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
      LUT_BIT_LEVEL_004 = 7,
      LUT_BIT_LEVEL_001 = 8
    } op_type;

    // set function type,
    op_type = LUT_BIT_LEVEL_001;

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
    case LUT_BIT_LEVEL_004:
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
    case LUT_BIT_LEVEL_001:
      {
	float variable[171] = {0.3203125, 0.328125 , 0.3359375, 0.34375  , 0.3515625, 0.359375 , 0.3671875,
			       0.375    , 0.3828125, 0.390625 , 0.3984375, 0.40625  , 0.4140625, 0.421875 ,
			       0.4296875, 0.4375   , 0.4453125, 0.453125 , 0.4609375, 0.46875  , 0.4765625,
			       0.484375 , 0.4921875, 0.5      , 0.5078125, 0.515625 , 0.5234375, 0.53125  ,
			       0.5390625, 0.546875 , 0.5546875, 0.5625   , 0.5703125, 0.578125 , 0.5859375,
			       0.59375  , 0.6015625, 0.609375 , 0.6171875, 0.625    , 0.6328125, 0.640625 ,
			       0.6484375, 0.65625  , 0.6640625, 0.671875 , 0.6796875, 0.6875   , 0.6953125,
			       0.703125 , 0.7109375, 0.71875  , 0.7265625, 0.734375 , 0.7421875, 0.75     ,
			       0.7578125, 0.765625 , 0.7734375, 0.78125  , 0.7890625, 0.796875 , 0.8046875,
			       0.8125   , 0.8203125, 0.828125 , 0.8359375, 0.84375  , 0.8515625, 0.859375 ,
			       0.8671875, 0.875    , 0.8828125, 0.890625 , 0.8984375, 0.90625  , 0.9140625,
			       0.921875 , 0.9296875, 0.9375   , 0.9453125, 0.953125 , 0.9609375, 0.96875  ,
			       0.9765625, 0.984375 , 0.9921875, 1.0      , 1.015625 , 1.03125  , 1.046875 ,
			       1.0625   , 1.078125 , 1.09375  , 1.109375 , 1.125    , 1.140625 , 1.15625  ,
			       1.171875 , 1.1875   , 1.203125 , 1.21875  , 1.234375 , 1.25     , 1.265625 ,
			       1.28125  , 1.296875 , 1.3125   , 1.328125 , 1.34375  , 1.359375 , 1.375    ,
			       1.390625 , 1.40625  , 1.421875 , 1.4375   , 1.453125 , 1.46875  , 1.484375 ,
			       1.5      , 1.515625 , 1.53125  , 1.546875 , 1.5625   , 1.578125 , 1.59375  ,
			       1.609375 , 1.625    , 1.640625 , 1.65625  , 1.671875 , 1.6875   , 1.703125 ,
			       1.71875  , 1.734375 , 1.75     , 1.765625 , 1.78125  , 1.796875 , 1.8125   ,
			       1.828125 , 1.84375  , 1.859375 , 1.875    , 1.890625 , 1.90625  , 1.921875 ,
			       1.9375   , 1.953125 , 1.96875  , 1.984375 , 2.0      , 2.125    , 2.25     ,
			       2.375    , 2.5      , 2.625    , 2.75     , 2.875    , 3.0      , 3.125    ,
			       3.25     , 3.375    , 3.5      , 3.625    , 3.75     , 3.875    , 4.0      ,
			       5.0      , 6.0      , 8.0 };
	float avg_value[171] = {0.3125        , 0.313312229604, 0.320340376412, 0.327333434597, 0.334290819118,
				0.34121195955 , 0.348096300198, 0.354943300194, 0.361752433588, 0.368523189421,
				0.375255071794, 0.38194759992 , 0.388600308168, 0.395212746092, 0.401784478454,
				0.408315085232, 0.414804161622, 0.421251318024, 0.427656180022, 0.434018388354,
				0.440337598868, 0.446613482474, 0.452845725083, 0.459034027536, 0.46517810553 ,
				0.471277689529, 0.477332524668, 0.483342370651, 0.489307001643, 0.495226206145,
				0.501099786875, 0.506927560627, 0.512709358142, 0.518445023952, 0.524134416232,
				0.52977740664 , 0.535373880157, 0.54092373491 , 0.546426882004, 0.55188324534 ,
				0.557292761429, 0.562655379207, 0.567971059841, 0.573239776531, 0.578461514313,
				0.583636269854, 0.588764051247, 0.593844877803, 0.598878779836, 0.603865798453,
				0.608805985338, 0.613699402534, 0.618546122223, 0.623346226508, 0.62809980719 ,
				0.632806965546, 0.637467812107, 0.642082466435, 0.646651056899, 0.651173720447,
				0.655650602392, 0.66008185618 , 0.66446764317 , 0.668808132415, 0.673103500437,
				0.677353931006, 0.681559614924, 0.685720749806, 0.689837539863, 0.693910195686,
				0.697938934035, 0.701923977626, 0.70586555492 , 0.709763899919, 0.713619251957,
				0.7174318555  , 0.721201959941, 0.724929819405, 0.728615692551, 0.732259842379,
				0.73586253604 , 0.739424044647, 0.742944643087, 0.746424609843, 0.749864226813,
				0.75326377913 , 0.756623554991, 0.759943845487, 0.764842817452, 0.771250232854,
				0.777505075554, 0.783609785076, 0.789566828323, 0.795378695007, 0.801047893278,
				0.806576945556, 0.811968384563, 0.817224749554, 0.822348582743, 0.827342425922,
				0.832208817272, 0.836950288352, 0.841569361283, 0.846068546092, 0.850450338243,
				0.854717216326, 0.858871639908, 0.862916047548, 0.866852854954, 0.870684453288,
				0.874413207618, 0.878041455492, 0.881571505649, 0.885005636853, 0.888346096843,
				0.891595101389, 0.894754833468, 0.89782744253 , 0.900815043864, 0.903719718056,
				0.906543510531, 0.909288431183, 0.911956454067, 0.914549517183, 0.917069522309,
				0.91951833491 , 0.921897784103, 0.924209662672, 0.926455727144, 0.928637697904,
				0.930757259365, 0.932816060166, 0.934815713419, 0.936757796986, 0.938643853791,
				0.940475392157, 0.942253886172, 0.943980776082, 0.945657468701, 0.947285337847,
				0.948865724786, 0.9503999387  , 0.951889257168, 0.953334926655, 0.954738163012,
				0.956100151989, 0.957422049751, 0.9587049834  , 0.959950051507, 0.96115832464 ,
				0.962330845898, 0.963468631449, 0.968098352301, 0.975066498368, 0.980527764768,
				0.984802089163, 0.988143811216, 0.990754196757, 0.992791947586, 0.994381858266,
				0.995621850551, 0.996588632293, 0.997342215536, 0.997929503203, 0.998387123973,
				0.998743665468, 0.999021428915, 0.999237805099, 0.999619      , 0.999948      ,
				0.999995};
	//0.910537, 0.920631, 0.929628, 0.937639, 0.944763, 0.951095, 0.956717, 0.961705,
	//0.979541, 0.997192, 0.999619, 0.999948, 0.999995};
	int length = sizeof(avg_value) / sizeof(avg_value[0]);

	float tanh_input = 0.0;
	for (int i = 0; i < N; ++i) {
	  tanh_input = fabs(input(i)) / 2;

	  if (tanh_input >= 0 && tanh_input < 0.3125)
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
