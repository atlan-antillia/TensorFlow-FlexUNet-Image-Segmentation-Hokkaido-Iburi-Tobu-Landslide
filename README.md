<h2>TensorFlow-FlexUNet-Image-Segmentation-Hokkaido-Iburi-Tobu-Landslide (2025/12/15)</h2>

Toshiyuki Arai<br>
Software Laboratory antillia.com<br>
<br>
This is the first experiment of Image Segmentation for <b>Hokkaido Iburi Tobu Landslide</b> (Singleclass) based on 
our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>)
, and a 512x512 pixels PNG 
<a href="https://drive.google.com/file/d/1Aym54XCS-dlbEtuENcnzc0zxOITxbCX0/view?usp=sharing">
<b>Augmented-Hokkaido-Iburi-Tobu-ImageMask-Dataset.zip </b></a> (3.3GB)
which was derived by us from <br><br>
<a href="https://zenodo.org/records/10294997/files/Hokkaido%20Iburi-Tobu.zip?download=1">Hokkaido Iburi-Tobu.zip </a> 
<br>
<a href="https://zenodo.org/records/10294997">
<b>CAS Landslide Dataset: A Large-Scale and Multisensor 
Dataset for Deep Learning-Based Landslide Detection</b>
</a>
<br><br>
<hr>
<b>Actual Image Segmentation for the Hokkaido-Iburi-Tobu Images of 512x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the 
our augmented dataset appear similar to the ground truth masks.<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/mini_test/images/10083.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/mini_test/masks/10083.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/mini_test_output/10083.png" width="320" height="auto"></td>
</tr>
</tr>
<td><img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/mini_test/images/10276.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/mini_test/masks/10276.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/mini_test_output/10276.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/mini_test/images/10483.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/mini_test/masks/10483.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/mini_test_output/10483.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
1 Dataset Citation
</h3>
The dataset used here was derived from <br><br>
<a href="https://zenodo.org/records/10294997/files/Hokkaido%20Iburi-Tobu.zip?download=1">Hokkaido Iburi-Tobu.zip </a> 
<br>
<a href="https://zenodo.org/records/10294997">
<b>CAS Landslide Dataset: A Large-Scale and Multisensor 
Dataset for Deep Learning-Based Landslide Detection</b>
</a>  on the zenodo web site.
<br><br>
In this work, we present the CAS Landslide Dataset, a large-scale and multisensor dataset for deep learning-based 
landslide detection, developed by the Artificial Intelligence Group at the Institute of Mountain Hazards 
and Environment, Chinese Academy of Sciences (CAS). The dataset aims to address the challenges encountered in 
landslide recognition. <br>
With the increase in landslide occurrences due to climate change and earthquakes, 
there is a growing need for a precise and comprehensive dataset to support fast and efficient 
landslide recognition. In contrast to existing datasets with dataset size, coverage, sensor type 
and resolution limitations, the CAS Landslide Dataset comprises 20,865 images, integrating satellite 
and unmanned aerial vehicle data from nine regions. <br>
 To ensure reliability and applicability, we establish a robust methodology to evaluate the dataset quality.<br> 
 We propose the use of the Landslide Dataset as a benchmark for the construction of landslide identification 
models and to facilitate the development of deep learning techniques. <br>
Researchers can leverage this dataset to obtain enhanced prediction, monitoring, and analysis capabilities, 
thereby advancing automated landslide detection.
<br><br>
<b>Citation</b><br>
Xu, Y., Ouyang, C., Xu, Q., Wang, D., Zhao, B., & Luo, Y. (2023). CAS Landslide Dataset:<br>
 A Large-Scale and Multisensor Dataset for Deep Learning-Based Landslide Detection [Data set].<br>
  Zenodo. https://doi.org/10.5281/zenodo.10294997
<br>
<br>
<b>License</b><br>
<a href="https://creativecommons.org/licenses/by-nc/4.0/legalcode">
Creative Commons Attribution Non Commercial 4.0 International
</a>
<br>
<br>
<h3>
2 Hokkaido-Iburi-Tobu ImageMask Dataset
</h3>
<h4>2.1 Download Hokkaido-Iburi-Tobu dataset</h4>
 If you would like to train this Hokkaido-Iburi-Tobu Segmentation model by yourself,
 please download the augmented <a href="https://drive.google.com/file/d/1Aym54XCS-dlbEtuENcnzc0zxOITxbCX0/view?usp=sharing">
 <b>Augmented-Hokkaido-Iburi-Tobu-ImageMask-Dataset.zip</b></a> (3.3GB) 
on the google drive, expand the downloaded, and put it under dataset folder to be:
<pre>
./Hokkaido-Iburi-Tobu
├─test
│  ├─images
│  └─masks
├─train
│  ├─images
│  └─masks
└─valid
    ├─images
    └─masks
</pre>
<b>Hokkaido-Iburi-Tobu Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/Hokkaido-Iburi-Tobu_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is large enough to use for a training set of our segmentation model.
<br><br> 
<h4>2.2 Hokkaido-Iburi-Tobu Dataset Derivation</h4>
The folder structure of the original 
<a href="https://www.kaggle.com/datasets/niyarrbarman/landslide-divided/data">
<b>Landslide Segmentation</b>
</a>
is the following.
<br><br>
<pre>
./Hokkaido Iburi-Tobu
 ├─img
 │  ├─Hokkaido0001.tif
 ...
 │  └─Hokkaido1571.tif
 ├─label
 │  ├─Hokkaido0001.tif
 ...
 │  └─Hokkaido1571.tif
 └─mask
     ├─Hokkaido0001.tif
 ...
     └─Hokkaido1571.tif
</pre>
We used the following 2 Python scripts to derive our 512x512 pixels PNG Augmented-Hokkaido-Iburi-Tobu dataset 
from <b>img</b> and <b>mask</b> folders containing 1,484 TIF files respectively.<br>
<ul>
<li><a href="./generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a></li>
<li><a href="./generator/split_master.py">split_master.py</a></li>
</ul>
<br>
<h4>2.3 Train Image and Mask samples </h4>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<br>
<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained Hokkaido-Iburi-Tobu TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu and, and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16</b> and a large <b>base_kernels = (11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = True

num_classes    = 2

base_filters   = 16
base_kernels   = (11,11)
num_layers     = 8
dropout_rate   = 0.05
dilation       = (1,1)
</pre>

<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.  
<pre>
[model]
model         = "TensorFlowFlexUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
You may specify other loss and metrics function names.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
<pre>
[mask]
mask_datatype    = "categorized"
mask_file_format = ".png"
;                    landslide: red
rgb_map = {(0,0,0):0, (255,0,0):1,}
</pre>


<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback (EpochChangeInferencer.py)</a></b>.<br>
<pre>
[train]
poch_change_infer     = True
epoch_change_infer_dir =  "./epoch_change_infer"
epoch_change_tiled_infer     = False
epoch_change_tiled_infer_dir =  "./epoch_change_tiled_infer"
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 28,29,30)</b><br>
<img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>

<b>Epoch_change_inference output at ending (epoch 58,59,60)</b><br>
<img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>


In this experiment, the training process was terminated at epoch 60.<br><br>
<img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/asset/train_console_output_at_epoch60.png" width="880" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/eval/train_metrics.png" width="520" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu</b> folder,<br>
and run the following bat file to evaluate TensorFlowUNet model for Hokkaido-Iburi-Tobu.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/asset/evaluate_console_output_at_epoch60.png" width="880" height="auto">
<br><br>Hokkaido-Iburi-Tobu
<a href="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this Hokkaido-Iburi-Tobu/test was not low, and dice_coef_multiclass not high as shown below.
<br>
<pre>
categorical_crossentropy,0.0476
dice_coef_multiclass,0.9743
</pre>
<br>
<h3>
5 Inference
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowFlexUNet model for Hokkaido-Iburi-Tobu.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
 
<img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for the Hokkaido-Iburi-Tobu Images of 512x512 pixels </b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the 
augmented dataset appear similar to the ground truth masks.<br>
<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<td><img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/mini_test/images/10110.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/mini_test/masks/10110.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/mini_test_output/10110.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/mini_test/images/10276.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/mini_test/masks/10276.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/mini_test_output/10276.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/mini_test/images/10364.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/mini_test/masks/10364.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/mini_test_output/10364.png" width="320" height="auto"></td>
</tr>


<tr>
<td><img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/mini_test/images/10483.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/mini_test/masks/10483.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/mini_test_output/10483.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/mini_test/images/10675.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/mini_test/masks/10675.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/mini_test_output/10675.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/mini_test/images/10784.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/mini_test/masks/10784.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hokkaido-Iburi-Tobu/mini_test_output/10784.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. Landslide4Sense: Multi-sensor landslide detection competition & benchmark dataset</b><br>
Institute of Advanced Research in Artificial Intelligence<br>
<a href="https://github.com/iarai/Landslide4Sense-2022">
https://github.com/iarai/Landslide4Sense-2022</a>
<br>
<br>
<b>2. The Outcome of the 2022 Landslide4Sense Competition:<br> Advanced Landslide Detection From Multisource Satellite Imagery
</b><br>
Omid Ghorbanzadeh; Yonghao Xu; Hengwei Zhao; Junjue Wang; Yanfei Zhong; Dong Zhao<br>
<a href="https://ieeexplore.ieee.org/document/9944085">
https://ieeexplore.ieee.org/document/9944085
</a>
<br>
<br>
<b>3. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
<b>4. TensorFlow-FlexUNet-Image-Segmentation-Landslide4Sense</b><br>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-Landslide4Sense">
https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-Landslide4Sense
</a>
<br>
<br>
<b>5. TensorFlow-FlexUNet-Image-Segmentation-Japan-Landslide</b><br>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-Japan-Landslide">
https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-Japan-Landslide
</a>

