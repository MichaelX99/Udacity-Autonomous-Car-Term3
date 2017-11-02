# Semantic Segmentation
## Running my code
'''
python main.py
'''
Change directory to tensorflow root
aka git clone https://github.com/tensorflow/tensorflow.git

'''
bazel build tensorflow/python/tools:optimize_for_inference
 
bazel-bin/tensorflow/python/tools/optimize_for_inference \
--input=${PATH_TO_PROJECT}/runs/frozen.pb \
--output=${PATH_TO_PROJECT}/runs/optimized.pb \
--frozen_graph=True \
--input_names="input_image, keep_prob" \
--output_names=logits

bazel build tensorflow/tools/graph_transforms:transform_graph

bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=${PATH_TO_PROJECT}/runs/optimized.pb \
--out_graph=${PATH_TO_PROJECT}/runs/eightbit_graph.pb \
--inputs="input_image, keep_prob" \
--outputs=logits \
--transforms='
add_default_attributes
remove_nodes(op=Identity, op=CheckNumerics)
fold_constants(ignore_errors=true)
fold_batch_norms
fold_old_batch_norms
fuse_resize_and_conv
quantize_weights
quantize_nodes
strip_unused_nodes
sort_by_execution_order'
'''

change back to the project directory

'''
python evaluate.py
'''

##Main Project Points
I did not use the load_vgg function that was provided for us, instead I used the generating vgg16 tf.slim code and manually extracted the weights from the saved vgg and loaded them into my own tensors.

Next I used tf.slim in order to eliminiate all the boilerplate code required for tensorflow.  

I then freeze the graph and provide the required scripts to optimize and quantize the trained model.

Finally there is a script for deployment on newly received images from a video stream.  It is a simple step forward to implement this into a ROS architecture and provide sudo real time performance on the Capstone project if required.

## Problem with Quantization
My performace dropped massively when I used the quantized version of my model and I believe that is because of my hardware and tensorflow setup.  Even though I have built tensorflow from source I imagine that in order to get the benefits of quantization you need to be on a jetson or a volta gpu but I could be wrong.


### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).
