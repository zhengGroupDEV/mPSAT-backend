'''
Author: rainyl
Description: utils
License: Apache License 2.0
'''
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, Union, List

INF = 0x3f3f3f

def splitDataset(N: int = INF, ptrain: float = 0.7, pval: float = 0.2, ptest: float = 0.1) -> None:
    """
        split dataset according to ratio
    :param N: the max number of img in one label
    """
    srcDir = Path("data/dataset/multi")
    dstDir = Path(f"data/dataset/datasetSplit{N}")

    if not dstDir.exists():
        dstDir.mkdir(parents=True)

    imgPaths = list(srcDir.glob("**/*.jpg"))
    np.random.shuffle(imgPaths)  # type: ignore

    imgPathN: List[Path] = []
    counts: Dict[str, int] = {}
    for path in imgPaths:
        mpid = path.parent.name
        counts[mpid] = counts[mpid] + 1 if mpid in counts else 1
        if counts[mpid] > N:
            # print(f"MPID {mpid} has more than {N} images, Skip...")
            continue
        imgPathN.append(path)

    np.random.shuffle(imgPathN)  # type: ignore

    ntrain = int(len(imgPathN) * ptrain)
    nval = int(len(imgPathN) * pval)

    imgTrain = imgPathN[:ntrain]
    imgVal = imgPathN[ntrain:ntrain+nval]
    imgTest = imgPathN[ntrain+nval:]


    def copyImg(imgs: List[Path], dstPath: Path):
        mpidCounts: Dict[str, int] = {}
        for imgPath in imgs:
            mpid = imgPath.parent.name
            mpidCounts[mpid] = mpidCounts[mpid] + 1 if mpid in mpidCounts else 0
            if mpidCounts[mpid] > N:
                print(f"MPID {mpid} has more than {N} images, Skip...")
                continue
            dst = dstPath / imgPath.parent.name / imgPath.name
            if not dst.parent.exists():
                dst.parent.mkdir(parents=True)
            if dst.exists():
                continue
            shutil.copy2(imgPath, dst)
            print("Copy {} to {}".format(imgPath, dst))

    copyImg(imgTrain, dstDir / "train")
    copyImg(imgVal, dstDir / "val")
    copyImg(imgTest, dstDir / "test")


if __name__ == "__main__":
    # DS430
    splitDataset(N=430, ptrain=0.7, pval=0.3, ptest=0.0)
    # DS1000
    splitDataset(N=1000, ptrain=0.7, pval=0.3, ptest=0.0)
