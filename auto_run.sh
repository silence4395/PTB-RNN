#!/bin/bash
#!/bin/sh
# time: 9/1/2017
# author: zhihui.luo@ingenic.com
#
#
####################################################

bit_width=(8 16 24)
fl_type1=(3 11 19)
fl_type2=(4 12 20)
origin_type=(LUT_BIT_LEVEL_004 LUT_BIT_LEVEL_001 DCT_LUT_6 DCT_LUT_6_PLUS AREAS)
approximate_type=(LUT_BIT_LEVEL_004_quan LUT_BIT_LEVEL_001_quan DCT_LUT_6_quan DCT_LUT_6_PLUS_quan AREAS_quan)
lstm_type=(tanh_diy sigmoid_diy sigmoid_tanh_diy)

lstm_length=${#lstm_type[@]}
app_length=${#approximate_type[@]}
bit_length=${#bit_width[@]}
lstm_index=0
app_index=0
bit_index=0

accuracy=100

function LookResult()
{
    grep "Test Perplexity:" log
    BINGO=$?
    if [ $BINGO -eq 0 ]
    then
	var=$(ps -ef | grep "Test Perplexity:" log)
	var=${var#*:}
	accuracy=${var%,*}
    else
	accuracy=NAN
    fi
    
    if [ $(echo "$accuracy == 100" | bc) -eq 1 ]
    then
	lstm_index=100
	app_index=100
	bit_index=100
	echo "ERROR: Please check your program."
    fi
}

rm -i result.txt
printf "%-20s %-25s %10s %10s %10s\n" LSTM_DIY_TYPE APPROXIMATE_TYPE BIT_WIDTH FRACTION ACCURACY>> result.txt
while(( $lstm_index<$lstm_length ))
do
    app_index=0
    while(( $app_index<$app_length ))
    do
	./run.sh ${lstm_type[$lstm_index]} ${origin_type[$app_index]} 32 23
	LookResult
	printf "%-20s %-30s %-10d %-10d %-3.10f\n" ${lstm_type[$lstm_index]} ${origin_type[$app_index]} 32 23 $accuracy>> result.txt
	if [ $app_index -gt 3 ]
	then
	    bit_index=0
	    while(( $bit_index<$bit_length ))
	    do
		./run.sh ${lstm_type[$lstm_index]} ${approximate_type[$app_index]} ${bit_width[$bit_index]} ${fl_type2[$bit_index]}
		LookResult
		printf "%-20s %-30s %-10d %-10d %-3.10f\n" ${lstm_type[$lstm_index]} ${approximate_type[$app_index]} ${bit_width[$bit_index]} ${fl_type2[$bit_index]} $accuracy >> result.txt
		let "bit_index++"
	    done
	    echo "=====================================" >> result.txt
	else
	    bit_index=0
	    while(( $bit_index<$bit_length ))
	    do
		./run.sh ${lstm_type[$lstm_index]} ${approximate_type[$app_index]} ${bit_width[$bit_index]} ${fl_type1[$bit_index]}
		LookResult
		printf "%-20s %-30s %-10d %-10d %-3.10f\n" ${lstm_type[$lstm_index]} ${approximate_type[$app_index]} ${bit_width[$bit_index]} ${fl_type1[$bit_index]} $accuracy >> result.txt
		let "bit_index++"
	    done
	    echo "=====================================" >> result.txt
	fi
	let "app_index++"
    done
    let "lstm_index++"
done