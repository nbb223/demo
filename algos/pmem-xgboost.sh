#/bin/bash

rm ./nohup.out

if [ "$1" -gt 50 ]; then
	echo "error: no larger than 50"
	exit
fi

#for idx in {1..10}; do
#nohup python3 4IntelDemo-xgboost.py /mnt/pmem1/ai_data/kaggle-creditcard/creditcard $idx & >> nohup.out
#done

idx=0
while [ "$idx" -le "$1" ]
do
#	nohup python3 4IntelDemo-xgboost.py /mnt/pmem1/ai_data/kaggle-creditcard/creditcard $idx & >> nohup.out
	echo $idx
	idx=$((idx+1))
done

