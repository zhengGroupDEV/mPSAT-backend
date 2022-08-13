#!/bin/bash

WORK_DIR=""
MODEL=$1
MODEL="${MODEL^^}"
FORMAT="csv"
SRC_DIR="data/MPe"
MODEL_FORMAT="skl"
cd $WORK_DIR

if [ "${MODEL}" == "DT" ];
then
    MODEL_PATH="repeatSave/dtSave"
    MODEL_NAME="clfDT.pkl"
elif [ "${MODEL}" == "RF" ];
then
    MODEL_PATH="repeatSave/rfSave"
    MODEL_NAME="clfRF.pkl"
elif [ "${MODEL}" == "LSVM" ]
then
    MODEL_PATH="repeatSave/svmSave"
    MODEL_NAME="clfLinearSVC.pkl"
elif [ "${MODEL}" == "CNN" ];
then
    MODEL_PATH="export_model/ds430"
    MODEL_NAME=""
    MODEL_FORMAT="onnx"
elif [ "${MODEL}" == "CNN_1000" ];
then
    MODEL_PATH="export_model/ds1000"
    MODEL_NAME=""
    MODEL_FORMAT="onnx"
else
    echo -e "MODEL $MODEL not supported, exit..."
    exit 1
fi

echo -e "Running Evaluating Environment Samples...\n\tmodel: $MODEL\n\tsrc_dir: $SRC_DIR\n\tmodel format: $MODEL_FORMAT"

CMD="python -m src.eval_environ_samples -m ${MODEL_FORMAT} -k 3 -f csv -s $SRC_DIR "

for i in {0..9}
do
    echo "Working on $i"
    m=$(echo -e "${MODEL}" | tr "[:upper:]" "[:lower:]")
    if [ ${MODEL} == "LSVM" ] || [ ${MODEL} == "DT" ] || [ ${MODEL} == "RF" ];
    then
        CMD1="${CMD} -p $MODEL_PATH/$(($i+1))/$MODEL_NAME -n eval_environ/ds430/${m}_${i}"
    elif [ ${MODEL} == "CNN" ];
    then
        CMD1="${CMD} -p $MODEL_PATH/${MODEL_NAME}${i}.onnx -n eval_environ/ds430/${m}_${i}"
    elif [ ${MODEL} == "CNN_1000" ];
    then
        CMD1="${CMD} -p $MODEL_PATH/${MODEL_NAME}${i}.onnx -n eval_environ/ds1000/${m}_${i}"
    fi
    # echo $CMD1
    r=`$CMD1`
    echo -e "$r"
done
