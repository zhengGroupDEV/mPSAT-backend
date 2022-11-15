$src_dir = "repeatSave/1000/dt"
# $src_dir = "repeatSave/cb-null/rf"
# $src_dir = "repeatSave/1000/rf"


$a = Get-ChildItem -Path $src_dir -Filter *.pkl -Recurse -ErrorAction SilentlyContinue -Force
foreach ($i in $a){
    $src = $i | ForEach-Object {$_.FullName}
    $dst = $src.Replace('.pkl', '.onnx')
    Write-Output "$src -> $dst"
    Write-Output ""
    python -m src.sk2onnx -s $src -d $dst
}
