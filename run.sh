#!/bin/bash
#!/bin/sh
# time: 7/25/2017
# author: zhihui.luo@ingenic.com
#
# Parameters config:
# Sigmoid optional value: SIGMOID PLAN SONF INTERPOLATION EXPONENT AREAS
# Tanh optional value: TANH PLAN EXPONENT AREAS
# LSTM activation type: origin sigmoid_diy tanh_diy sigmoid_tanh_diy
#
####################################################

sigmoid_type=AREAS
tanh_type=AREAS
lstm_type=sigmoid_tanh_diy

function SetSigmoidType()
{
    awk -v type=$1 -F ' ' '{ if (($1 == "op_type") && ($2 == "="))
                   { print " "" "" "" " $1 " " $2 " " type";"}
                   else { print $0;}}' sigmoid_diy.cpp >| tmp.cpp
    cp tmp.cpp sigmoid_diy.cpp
    rm tmp.cpp
}

function SetTanhType()
{
    awk -v type=$1 -F ' ' '{ if (($1 == "op_type") && ($2 == "="))
                   { print " "" "" "" " $1 " " $2 " " type";"}
                   else { print $0;}}' tanh_diy.cpp >| tmp.cpp
    cp tmp.cpp tanh_diy.cpp
    rm tmp.cpp
}

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
cd util/usr_op
SetSigmoidType $sigmoid_type
./compile.sh
echo "Sigmoid function type:"
grep "op_type ="  sigmoid_diy.cpp 
echo "Sigmoid compiler done!"

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
cd tanh/
SetTanhType $tanh_type
./compile.sh
echo "Tanh function type:"
grep "op_type =" tanh_diy.cpp
echo "Tanh compiler done!"

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
cd ../../../
./script/ptb_test.py --lstm_type=$lstm_type
