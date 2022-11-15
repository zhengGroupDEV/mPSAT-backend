$topk = 10
$ds = "data/ds_mpe_v1.2.ftr"
$mpid_label = "data/mpid_label.json"
$label_name = "F:/opt/dataset/ds_300_3600_cb_nrpb/full/label_name.json"
$device = "cpu"

##### CNN #####
$bs = 100
$model_name = "cnn"
$work_home = "repeatSave/1000"
# $work_home = "repeatSave/final"

# ##### CNN2D #####
# $bs = 100
# $model_name = "cnn2d"
# $work_home = "repeatSave/1000"
# $work_home = "repeatSave/final"

# ##### DT #####
# $bs = 100
# $model_name = "dt"
# # $work_home = "repeatSave/final"
# $work_home = "repeatSave/1000"

##### RF #####
# $bs = 100
# $model_name = "rf"
# # $work_home = "repeatSave/final"
# $work_home = "repeatSave/1000"


foreach ($i in $0..9) {
    $model = "$work_home/$model_name/$i/clf_$model_name.onnx"
    $saveto = "$work_home/$model_name/$i"
    infer_one $model_name $model $saveto $bs
}

function infer_one ($model_name_, $model_path_, $saveto_, $bs_ = $bs) {
    Write-Output "****************************************"
    Write-Output "model: $model_name_"
    Write-Output "model_path: $model_path_"
    Write-Output "saveto: $saveto_"
    Write-Output "****************************************"
    Write-Output ""

    python -m src.infer `
        --eval-ds -k $topk `
        --device $device `
        --dataset-path $ds `
        --label-name $label_name `
        --mpid-label $mpid_label `
        --model-name $model_name_ `
        -m $model_path_ `
        --saveto $saveto_ `
        --batch-size $bs_
}
