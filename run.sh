#!/bin/bash
#!/bin/sh
# time: 7/25/2017
# author: zhihui.luo@ingenic.com
#
# Parameters config:
# Sigmoid optional value:
#     SIGMOID PLAN SONF INTERPOLATION EXPONENT AREAS PLAN_LUT LUT_BIT_LEVEL_004 LUT_BIT_LEVEL_001
# Tanh optional value:
#     TANH PLAN EXPONENT AREAS PLAN_LUT LUT_BIT_LEVEL_004 LUT_BIT_LEVEL_001 DCT_LUT_6 DCT_LUT_4
#     LUT_BIT_LEVEL_004_quan
# LSTM activation type:
#     origin sigmoid_diy tanh_diy sigmoid_tanh_diy
#
####################################################
lstm_type=$1

sigmoid_type=$2
tanh_type=$2

bit_width=$3
fraction_length=$4

# function for change sigmoid type
function SetSigmoidType()
{
    awk -v type=$1 -F ' ' '{ if (($1 == "op_type") && ($2 == "="))
                   { print " "" "" "" " $1 " " $2 " " type";"}
                   else { print $0;}}' sigmoid_diy.cpp >| tmp.cpp
    cp tmp.cpp sigmoid_diy.cpp
    rm tmp.cpp
}

# function for change tanh type
function SetTanhType()
{
    awk -v type=$1 -F ' ' '{ if (($1 == "op_type") && ($2 == "="))
                   { print " "" "" "" " $1 " " $2 " " type";"}
                   else { print $0;}}' tanh_diy.cpp >| tmp.cpp
    cp tmp.cpp tanh_diy.cpp
    rm tmp.cpp
}

# function for set tanh fixed-point bit-width
function SetTanhBitWidth()
{
    awk -v bw=$1 -v fl=$2 -F ' ' '{ if (($2 == "const") && ($3 == "int") && ($4 == "bit_width"))
                                    {print " " " " $1 " " $2 " " $3 " " $4 " " $5 " " bw ";"}
                                 else if (($2 == "const") && ($3 == "int") && ($4 == "fl"))
                                    {print " " " " $1 " " $2 " " $3 " " $4 " " $5 " " fl ";"}
                                 else
                                    {print $0;}}' tanh_diy.cpp >| tmp.cpp
    cp tmp.cpp tanh_diy.cpp
    rm tmp.cpp
}
# function for set sigmoid fixed-point fraction-length
function SetSigmoidBitWidth()
{
    awk -v bw=$1 -v fl=$2 -F ' ' '{if (($2 == "const") && ($3 == "int") && ($4 == "bit_width"))
                                    {print " " " " $1 " " $2 " " $3 " " $4 " " $5 " " bw ";"}
                                 else if (($2 == "const") && ($3 == "int") && ($4 == "fl"))
                                    {print " " " " $1 " " $2 " " $3 " " $4 " " $5 " " fl ";"}
                                 else
                                    {print $0;}}' sigmoid_diy.cpp >| tmp.cpp
    cp tmp.cpp sigmoid_diy.cpp
    rm tmp.cpp
}

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
cd activation_util/sigmoid
SetSigmoidType $sigmoid_type
SetSigmoidBitWidth $bit_width $fraction_length
./compile.sh |& tee log

# check compile
grep "error" log
ERROR=$?
if [ $ERROR -eq 0 ]; then
    exit
fi

echo "Sigmoid function type:"
grep "op_type ="  sigmoid_diy.cpp
echo "Sigmoid fixed point info:"
grep "static const int bit_width =" sigmoid_diy.cpp
grep "static const int fl =" sigmoid_diy.cpp
echo "Sigmoid compiler done!"

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
cd ../tanh/
SetTanhType $tanh_type
SetTanhBitWidth $bit_width $fraction_length
./compile.sh |& tee log

# check compile
grep "error" log
ERROR=$?
if [ $ERROR -eq 0 ]; then
    exit
fi

echo "Tanh function type:"
grep "op_type =" tanh_diy.cpp
echo "Sigmoid fixed point info:"
grep "static const int bit_width =" tanh_diy.cpp
grep "static const int fl =" tanh_diy.cpp
echo "Tanh compiler done!"

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "LSTM mode: $lstm_type"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

cd ../../
./script/ptb_test.py --lstm_type=$lstm_type |& tee log
