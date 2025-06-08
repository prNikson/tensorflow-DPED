
## DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks

#### 1. Prerequisites
- Python 3.11
- uv package manager
- TensorFlow 2.19
- Nvidia GPU

#### 3. First steps
1. Clone this repository
```bash
git clone https://github.com/prNikson/tensorflow-DPED
```
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation) to your system
3. Create a virtual environment and install the required packages:
```bash
cd tensorflow-DPED/
uv sync
```
4. Log in to [Weights & Biases (wandb)](https://wandb.ai/) to monitor your training progress
5. Download VGG-19 model
6. Download basic dataset from the authors on request 
- Download the pre-trained [VGG-19 model](https://polybox.ethz.ch/index.php/s/7z5bHNg5r5a0g7k) <sup>[Mirror](https://drive.google.com/file/d/0BwOLOmqkYj-jMGRwaUR2UjhSNDQ/view?usp=sharing&resourcekey=0-Ff-0HUQsoKJxZ84trhsHpA)</sup> and put it into `vgg_pretrained/` folder
- Download [DPED dataset](http://people.ee.ethz.ch/~ihnatova/#dataset) (patches for CNN training) and extract it into `dped/` folder.  
<sub>This folder should contain three subolders: `sony/`, `iphone/` and `blackberry/`</sub>
7. Optionally, you can use our custom dataset by placing it in the `kvadra/` folder
Link for download: https://huggingface.co/datasets/i44p/dped-pytorch/tree/main
training sample of photos: `train.tar.zst`
testing sample of photos: `test.tar.zst`
patches from the training sample (~300k patches): `train_patches.tar.zst`
patches from the testing sample (~5k patches): `test_patches.tar.zst`
full dataset (~1200 photos): `full_dataset_jpeg.tar.zst`
Extract `train_patches.tar.zst` and `test_patches.tar.zst` into `dped/kvadra/training_data/` and `dped/kvadra/test_data/patches/`, respectively.  
Rename folder `target` to `canon` and folder `input` to `kvadra`.

#### 4. Train the model
```bash
uv run train_model.py model=<model> batch_size=<batch_size>
```

Obligatory parameters:

>`model`: **`iphone`**, **`blackberry`**, **`sony`**, **`kvadra`**

Optional parameters and their default values:

>```batch_size```: **```50```** &nbsp; - &nbsp; batch size [smaller values can lead to unstable training] <br/>
>```train_size```: **```30000```** &nbsp; - &nbsp; the number of training patches randomly loaded each ```eval_step``` iterations. You can also load the entire dataset instead of just 30,000 patches from it each `eval_step`. To do this, enter `train_size=-1` in command line. But for this you need have a lot of memory<br/>
>```eval_step```: **```1000```** &nbsp; - &nbsp; each ```eval_step``` iterations the model is saved and the training data is reloaded <br/>
>```num_train_iters```: **```20000```** &nbsp; - &nbsp; the number of training iterations <br/>
>```learning_rate```: **```5e-4```** &nbsp; - &nbsp; learning rate <br/>
>```w_content```: **```10```** &nbsp; - &nbsp; the weight of the content loss <br/>
>```w_color```: **```0.5```** &nbsp; - &nbsp; the weight of the color loss <br/>
>```w_texture```: **```1```** &nbsp; - &nbsp; the weight of the texture [adversarial] loss <br/>
>```w_tv```: **```2000```** &nbsp; - &nbsp; the weight of the total variation loss <br/>
>```dped_dir```: **```dped/```** &nbsp; - &nbsp; path to the folder with DPED dataset <br/>
>```vgg_dir```: **```vgg_pretrained/imagenet-vgg-verydeep-19.mat```** &nbsp; - &nbsp; path to the pre-trained VGG-19 network <br/>
>```gpu```: &nbsp; - &nbsp; gpu number if you have several (0 by default)<br/>

Example:

```bash
uv run train_model.py model=kvadra batch_size=50 dped_dir=dped/
```

#### 6. Test the obtained models

```bash
uv test_model.py model=<model>
```

Obligatory parameters:

>```model```: **```iphone```**, **```blackberry```** , **```sony```**, **`kvadra`**

Optional parameters:

>```test_subset```: **```full```**,**```small```**  &nbsp; - &nbsp; all 29 or only 5 test images will be processed <br/>
>```iteration```: **```all```** or **```<number>```**  &nbsp; - &nbsp; get visual results for all iterations or for the specific iteration,  
>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**```<number>```** must be a multiple of ```eval_step``` <br/>
>```resolution```: **```orig```**,**```high```**,**```medium```**,**```small```**,**```tiny```** &nbsp; - &nbsp; the resolution of the test 
images [**```orig```** means original resolution]<br/>
>```use_gpu```: **```true```**,**```false```** &nbsp; - &nbsp; run models on GPU or CPU <br/>
>```dped_dir```: **```dped/```** &nbsp; - &nbsp; path to the folder with DPED dataset <br/>  

Example:

```bash
uv run test_model.py model=kvadra iteration=19000 test_subset=full resolution=orig use_gpu=true 
```
For high-resolution images (e.g., 4224×3136), TensorFlow may fail to allocate memory. In that case, use `use_gpu=false`.
This script processes photos from:  
`dped/kvadra/test_data/full_size_test_images/`
To process a single image, use `test_image.py`.
Kvadra model  supports photo size 4224x3136. All resolutions are specified in `utils.py`
Example:
```bash
uv run test_image.py <path_to_image> --iter 19000 --gpu true
```
iteration=19000 by default
gpu=false by default
<br/>

#### 7. Folder structure

>```dped/```              &nbsp; - &nbsp; the folder with the DPED dataset <br/>
>```models/```            &nbsp; - &nbsp; logs and models that are saved during the training process <br/>
>```models_orig/```       &nbsp; - &nbsp; the provided pre-trained models for **```iphone```**, **```sony```** and **```blackberry```** <br/>
>```results/```           &nbsp; - &nbsp; visual results for small image patches that are saved while training <br/>
>```vgg-pretrained/```    &nbsp; - &nbsp; the folder with the pre-trained VGG-19 network <br/>
>```visual_results/```    &nbsp; - &nbsp; processed [enhanced] test images <br/>

>```load_dataset.py```    &nbsp; - &nbsp; python script that loads training data <br/>
>```models.py```          &nbsp; - &nbsp; architecture of the image enhancement [resnet] and adversarial networks <br/>
>```ssim.py```            &nbsp; - &nbsp; implementation of the ssim score <br/>
>```train_model.py```     &nbsp; - &nbsp; implementation of the training procedure <br/>
>```test_model.py```      &nbsp; - &nbsp; applying the pre-trained models to test images <br/>
>```utils.py```           &nbsp; - &nbsp; auxiliary functions <br/>
>```vgg.py```             &nbsp; - &nbsp; loading the pre-trained vgg-19 network <br/>

