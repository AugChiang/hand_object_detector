# Hand Object Detector
This is the code for our paper *Understanding Human Hands in Contact at Internet Scale* (CVPR 2020, **Oral**).

Dandan Shan, Jiaqi Geng*, Michelle Shu*, David F. Fouhey

![method](assets/method.png)



## Introduction

This repo is the pytorch implementation of our Hand Object Detector based on Faster-RCNN.

More information can be found at our:

* [Project and dataset webpage](http://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/)


## Prerequisites

The recommended setup uses Python 3.12, PyTorch 2.7.1, Torchvision 0.22.1, and CUDA 12.8. The demo and extension build were verified on Linux with PyTorch 2.7.1 (`cu126`) and the CUDA 12.8 Toolkit (`nvcc`). The CUDA toolkit is needed to compile the custom ROI and NMS operators; the NVIDIA driver must support the selected CUDA runtime. PyTorch also provides matching `cu128` wheels, installed with the command below.

Create and activate a Conda environment, then install the CUDA 12.8 PyTorch wheels:

```bash
conda create --name handobj python=3.12 pip
conda activate handobj
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

The `requirements.txt` file contains the remaining Python libraries and intentionally leaves PyTorch/Torchvision to the command above so pip does not replace the CUDA-enabled wheels with a different build.


## Preparation

First, clone this fork:
```
git clone https://github.com/AugChiang/hand_object_detector && cd hand_object_detector
```


## Environment & Compilation

Install the remaining Python dependencies:
```
python -m pip install -r requirements.txt
```

Compile and place the custom CUDA/C++ extension in the source package:
```
cd lib
python setup.py build_ext --inplace
cd ..
```

Build the extension from an environment where the GPU is visible and `nvcc` from CUDA 12.8 is on `PATH`. This command builds in place, which works with the repository's `_init_paths.py` and avoids an isolated editable install that cannot find PyTorch during extension compilation. Re-run it after changing PyTorch, CUDA, or the extension sources.

Check the installed PyTorch CUDA runtime and compiler before building:
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvcc --version
```

To run the included image demo after placing a checkpoint in the directory layout described in the Demo section:
```bash
python demo.py --cuda --checksession 1 --checkepoch 8 --checkpoint 89999
```

<!-- You will meet some errors about coco dataset: (not the best but the easiest solution)
```
cd data
git clone https://github.com/pdollar/coco.git 
cd coco/PythonAPI
make
``` -->
<!-- 
If you meet some error about spicy, make sure you downgrade to scipy=1.1.0:
```
pip install scipy=1.1.0
``` -->

PS:

Since the repo is modified based on [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0) (use branch pytorch-1.0), if you have futher questions about the environments, the issues in that repo may help.

## Performance (AP)
<!-- Table, test on all -->
- Tested on the testset of our **100K and ego** dataset:
<table><tbody>
<tr>
<td align="center">Name</td>
<td align="center">Hand</td>
<td align="center">Obj</td>
<td align="center">H+Side</td>
<td align="center">H+State</td>
<td align="center">H+O</td>
<td align="center">All</td>
<td align="center">Model Download Link</td>
</tr>

<tr>
<td align='left'>handobj_100K+ego</td>
<td align='center'>90.4</td>
<td align='center'>66.3</td>
<td align='center'>88.4</td>
<td align='center'>73.2</td>
<td align='center'>47.6</td>
<td align='center'>39.8</td>
<td align="center"><a href="https://drive.google.com/open?id=1H2tWsZkS7tDF8q1-jdjx6V9XrK25EDbE">faster_rcnn_1_8_132028.pth</a></td>
</tr>

<tr>
<td align='left'>handobj_100K</td>
<td align='center'>89.8</td>
<td align='center'>51.5</td>
<td align='center'>65.8</td>
<td align='center'>62.4</td>
<td align='center'>27.9</td>
<td align='center'>20.9</td>
<td align="center"><a href="https://drive.google.com/open?id=166IM6CXA32f9L6V7-EMd9m8gin6TFpim">faster_rcnn_1_8_89999.pth</a></td>
</tr>

</tbody></table>



<!-- Table, test on 100K -->
- Tested on the testset of our **100K** dataset:
<table><tbody>
<tr>
<tr><td align="center">Name</td>
<td align="center">Hand</td>
<td align="center">Obj</td>
<td align="center">H+Side</td>
<td align="center">H+State</td>
<td align="center">H+O</td>
<td align="center">All</td>
</tr>

<tr>
<td align='left'>handobj_100K+ego</td>
<td align='center'>89.6</td>
<td align='center'>64.7</td>
<td align='center'>79.0</td>
<td align='center'>63.8</td>
<td align='center'>45.1</td>
<td align='center'>36.8</td>
</tr>

<tr>
<td align='left'>handobj_100K</td>
<td align='center'>89.6</td>
<td align='center'>64.0</td>
<td align='center'>78.9</td>
<td align='center'>64.2</td>
<td align='center'>46.9</td>
<td align='center'>38.6</td>
</tr>

</tbody></table>


<!-- Table, test on ego -->
- Tested on the testset of our **ego** dataset:
<table><tbody>
<tr>
<tr><td align="center">Name</td>
<td align="center">Hand</td>
<td align="center">Obj</td>
<td align="center">H+Side</td>
<td align="center">H+State</td>
<td align="center">H+O</td>
<td align="center">All</td>
</tr>

<tr>
<td align='left'>handobj_100K+ego</td>
<td align='center'>90.5</td>
<td align='center'>67.2</td>
<td align='center'>90.0</td>
<td align='center'>75.0</td>
<td align='center'>47.4</td>
<td align='center'>46.3</td>
</tr>

<tr>
<td align='left'>handobj_100K</td>
<td align='center'>89.8</td>
<td align='center'>41.7</td>
<td align='center'>59.5</td>
<td align='center'>62.8</td>
<td align='center'>20.3</td>
<td align='center'>12.7</td>
</tr>

</tbody></table>


The model **handobj_100K** is trained on trainset of **100K** youtube frames. 

The model **handobj_100K+ego** is trained on trainset of **100K** plus additional **egocentric** data we annotated, which works much better on egocentric data. 

We provide the frame names of the egocentric data we used here: [trainval.txt](https://github.com/ddshan/hand_object_detector/blob/master/assets/data_ego_framename/trainval.txt), [test.txt](https://github.com/ddshan/hand_object_detector/blob/master/assets/data_ego_framename/test.txt). This split is backwards compatible with
the [Epic-Kitchens2018](https://epic-kitchens.github.io/2018) (EK), [EGTEA](http://cbs.ic.gatech.edu/fpv/), and [CharadesEgo](https://prior.allenai.org/projects/charades-ego) (CE).



## Train

### Data preparation  
Prepare and save pascal-voc format data in **data/** folder:
```
mkdir data
```
You can download our prepared pascal-voc format data from [pascal_voc_format.zip](https://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/downloads/pascal_voc_format.zip) (see more of our downloads on our [project and dataset webpage](http://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/download.html)).


### Download pre-trained Resnet-101 model
Download pretrained Resnet-101 model from [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0) (go to **Pretrained Model** section, download **ResNet101** from their Dropbox link) and save it like:
```
data/pretrained_model/resnet101_caffe.pth
```

So far, the data/ folder should be like this:
```
data/
├── pretrained_model
│   └── resnet101_caffe.pth
├── VOCdevkit2007_handobj_100K
│   └── VOC2007
│       ├── Annotations
│       │   └── *.xml
│       ├── ImageSets
│       │   └── Main
│       │       └── *.txt
│       └── JPEGImages
│           └── *.jpg
```

To train a hand object detector model with resnet101 on pascal_voc format data, run:
```
CUDA_VISIBLE_DEVICES=0 python trainval_net.py --model_name handobj_100K --log_name=handobj_100K --dataset pascal_voc --net res101 --bs 1 --nw 4 --lr 1e-3 --lr_decay_step 3 --cuda --epoch=10 --use_tfb 
```



## Test
To evaluate the detection performance, run:
```
CUDA_VISIBLE_DEVICES=0 python test_net.py --model_name=handobj_100K --save_name=handobj_100K --cuda --checkepoch=xxx --checkpoint=xxx
```


## Demo

### Image Demo

**Download models** by using the links in the table above from google drive.


**Save models** in the **models/** folder:
```
mkdir models

models
└── res101_handobj_100K
    └── pascal_voc
        └── faster_rcnn_{checksession}_{checkepoch}_{checkpoint}.pth
```



**Simple testing**: 

Put your images in the **images/** folder and run the command. A new folder **images_det** will be created with the visualization. Check more about argparse parameters in demo.py.
```
CUDA_VISIBLE_DEVICES=0 python demo.py --cuda --checkepoch=xxx --checkpoint=xxx
```

### Video Inference

`inference.py` runs Faster R-CNN on video frames in batches and writes an annotated video. Download a checkpoint from the model table above and place it in the checkpoint directory layout shown above. For example, with `faster_rcnn_1_8_89999.pth` at `models/res101_handobj_100K/pascal_voc/`, run:

```bash
python inference.py --video path/to/video.mp4 --checkpoint 89999 --cuda --batch_size 4
```

The output is `video_det.mp4` by default. Use `--output path/to/output.mp4` to choose another path. `--batch_size` (or `--bs`) sets how many frames are processed in one model inference; the final group may contain fewer frames. For CPU inference, omit `--cuda`.


### Detection arrays

The output arrays follow the hand and object predictions described in Section 4.1 of the [100DOH paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Shan_Understanding_Human_Hands_in_Contact_at_Internet_Scale_CVPR_2020_paper.html). Each array has shape `(N, 10)`: one row per detection that remains after the confidence threshold and non-maximum suppression. `hand_dets` contains hand detections; `obj_dets` contains contacted-object box detections. A row in one array is **not** paired with the row at the same index in the other array.

The ten columns in each row are:

| Columns | Field | Meaning |
| --- | --- | --- |
| `0:4` | `boxes` | `[x1, y1, x2, y2]` in pixels in the original input image. `(x1, y1)` is the top-left corner and `(x2, y2)` is the bottom-right corner; this is not `[x, y, width, height]`. |
| `4` | `score` | Confidence for the detection’s class, from 0 to 1. |
| `5` | `contact_state` | Integer class ID describing what the detected hand is touching (or not touching): `0` no contact; `1` self-contact; `2` contact with another person; `3` contact with a portable object; `4` contact with a stationary/non-portable object. |
| `6:9` | `offset_vector` | `[m, vx, vy]`: a hand-to-object association vector, factored into magnitude `m` and unit direction `(vx, vy)`. In this implementation, the object-center estimate is `hand_box_center + 10000 * m * (vx, vy)` in image pixels. |
| `9` | `left/right` | Hand side: `0` left, `1` right. |

The `state`, `offset_vector`, and `left/right` fields in `obj_dets` are included to keep both arrays the same width; they are not trained for objects and should be ignored. Use only the object box and score from `obj_dets`. The auxiliary fields in `hand_dets` carry the hand’s contact state, the associated-object direction, and its side.

In plain terms, **self-contact** means the hand touches the same person’s body; **another person** means it touches someone else; a **portable object** can be moved (for example, a cup); and a **stationary/non-portable object** is fixed in place (for example, furniture). The contact-state value is not an object category or an action label. When `--print_detections` is enabled, hand state IDs are printed with their meanings; object state predictions are untrained and marked to be ignored.

To print these arrays for each image while running the demo, pass `--print_detections`:
```bash
python demo.py --cuda --checksession 1 --checkepoch 8 --checkpoint 89999 --print_detections
```
Each row is one detection with fields in the order listed above; missing detections are printed as `None`.

**Matching**:

Check the additional [matching.py](https://github.com/ddshan/Hand_Object_Detector/blob/master/lib/model/utils/matching.py) script to match the detection results, **hand_dets** and **obj_dets**, if needed.  


### One Image Demo Output:

Color definitions:
* yellow: object bbox
* red: right hand bbox
* blue: left hand bbox

Label definitions:
* L: left hand
* R: right hand
* N: no contact
* S: self contact
* O: other person contact
* P: portable object contact
* F: stationary object contact (e.g.furniture)


![demo_sample](assets/boardgame_848_sU8S98MT1Mo_00013957.png)


### Limitations
- Occasional false positives with no people.
- Issues with left/right in egocentric data (Please check egocentric models that work far better).
- Difficulty parsing the full state with lots of people.

<!-- ## Acknowledgment

xxx -->

## Citation

If this work is helpful in your research, please cite:
```
@INPROCEEDINGS{Shan20, 
    author = {Shan, Dandan and Geng, Jiaqi and Shu, Michelle  and Fouhey, David},
    title = {Understanding Human Hands in Contact at Internet Scale},
    booktitle = CVPR, 
    year = {2020} 
}
```
When you use the model trained on our ego data, make sure to also cite the original datasets ([Epic-Kitchens](https://epic-kitchens.github.io/2018), [EGTEA](http://cbs.ic.gatech.edu/fpv/) and [CharadesEgo](https://prior.allenai.org/projects/charades-ego)) that we collect from and agree to the original conditions for using that data.
