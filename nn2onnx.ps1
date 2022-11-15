# $src_dir = Get-Item "repeatSave/final/cnn"
$src_dir = Get-Item "repeatSave/1000/cnn"
$model_name = "cnn"

# $src_dir = Get-Item "repeatSave/final/cnn2d"
# $src_dir = Get-Item "repeatSave/1000/cnn2d"
# $model_name = "cnn2d"

$model_name = $src_dir | ForEach-Object {$_.name}

$a = Get-ChildItem -Path $src_dir -Filter *.ckpt -Recurse -ErrorAction SilentlyContinue -Force
foreach ($i in $a){
    $src = $i | ForEach-Object {$_.FullName}
    $name = $i | ForEach-Object {$_.name}

    Write-Output "#####################################################"
    $dst = $src.Replace($name, "clf_$model_name.onnx")
    Write-Output "$src -> $dst"
    python -m src.nn2onnx -s $src -d $dst -m $model_name
    Write-Output ""
    Write-Output "#####################################################"
}

