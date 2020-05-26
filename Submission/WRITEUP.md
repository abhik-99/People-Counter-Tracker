# Project Write-Up

### Project People Counter App
**BY** - Abhik Banerjee

**Contact** - abhik@abhikbanerjee.com, abhik.banerjee.1999@gmail.com

**Model Used** - Single-shot detector V1 trained on CoCo Dataset

**Link to Test Video** - https://youtu.be/aMronQi4H4I

**STEPS to Reproduce** -
1. Source the Environment.
2. export DOWNLOADER_PATH="/opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/"
3. $DOWNLOADER_PATH/downloader.py --name ssd_mobilenet_v1_coco (Public Model from Open Model Zoo).
4. mkdir IR
5. python mo.py --input_model public/ssd_mobilenet_v1_coco/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config public/ssd_mobilenet_v1_coco/ssd_mobilenet_v1_coco_2018_01_28/pipeline.config --tensorflow_use_custom_operations_config $MO/extensions/front/tf/ssd_support.json --output_dir IR
6. Follow steps in Page 4 of Guide.
5. python main.py -m IR/ssd_mobilenet_v1_coco/frozen_inference_graph.xml -i resources/Pedestrian_Detect_2_1_1.mp4 -d CPU -l /opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

**NOTE** 
An output file titled "output.mp4" is produced after the whole stream concludes. In case of image, "output_image.jpg" is generated in the project root.

## 1. Explaining Custom Layers
Custom Layers are those layer which are not supported by default on Intel OpenVINO. These layers include but are not limited to- any pre/post processing steps incorporated in the model, any un-supported operation in the model.
The most common way to tackle such a problem can be to offload the computation of that layer to CPU. This can also involve cutting up the model till before the unsupported layer and then after that layer to the end.
The models chosen for this specific task were - SSD V1, SSD v2 and SSD300. The models were trained on Microsoft's Coco Dataset. No custom layers were detected in the models used. Utlimately, SSD v1 was used for the project.

## 2. Comparing Model Performance

For reasoning behind choosing SSD Mobilenet v1 among SSD Mobilenet v1, v2 and SSD300, please skip to the section **5**.

My method(s) to compare model before and after conversion to Intermediate Representations
were:- 
1. The Model was passed through Model Optimizer and IR was generated. These IRs were loaded into the Inference Engine and inference was done.

A .pbtext file could not be generated using OpenCV's tf_text_graph_ssd.py (https://github.com/opencv/opencv/tree/master/samples/dnn) due to Error in Importing Tensorflow error as sumarized by:

**"""

Traceback (most recent call last):
  File "tf_text_graph_ssd.py", line 405, in <module>
    createSSDGraph(args.input, args.config, args.output)
  File "tf_text_graph_ssd.py", line 128, in createSSDGraph
    writeTextGraph(modelPath, outputPath, outNames)
  File "/home/workspace/opencv/samples/dnn/tf_text_graph_common.py", line 316, in writeTextGraph
    from tensorflow.tools.graph_transforms import TransformGraph
ModuleNotFoundError: No module named 'tensorflow.tools.graph_transforms'
    
"""**

Steps to reproduce the error:-

1. pip install tensorflow
2. git clone https://github.com/opencv/opencv.git
3. cd opencv/samples/dnn/
4. mkdir exported_pbtext
5. python tf_text_graph_ssd.py --input /home/workspace/public/ssd_mobilenet_v1_coco/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb --config /home/workspace/public/ssd_mobilenet_v1_coco/ssd_mobilenet_v1_coco_2018_01_28/pipeline.config --output exported_pbtext/ssd_v1.pbtext

An issue regarding the same would be raised in the OpenCV Repo (hosted at the URL mentioned in step 2)after the current project review has concluded.
    

## 3. Assess Model Use Cases

From the very Video that was given for usage, I can think of one immediate use case:-
1. Tracking the behaviour of a person in a given area of interest. The People in the video seemed to enter the frame, read from a piece of paper and then leave from one specific side. The Model can be used to predict if the person takes the wrong step - this can be detected via tracking the position of the centroid.
2. The model can be used to assess the time a person spends in the frame. 

These two points can also be commonly observed in *ATMs* were at a given time only 1 person should be present in the frame and they should not be present more than a specific alloted time. 

Centroid tracking was used in the project. People who were actively being tracked were marked with a green centroid while those who have exited the frame were denoted by a red centroid at the last point they were spotted. This specific feature can be used for trajectory as well as most favoured point of exit prediction.

## 4. Assess Effects on End User Needs

The End User may need speed and this can be brought at the cost of accuracy. However, the end user can also get a measure of how many people are in the frame and not their exact location. This can help reduce the total inference + post-processing time.
If the end user needs accuracy, it can affect the speed of inference + post-processing. However, trackers like Kalman Filters can be used to track the person and prevent inferencing on every frame. This would significantly reduce inference time without affecting accuracy.

## 5. Model Research

All 3 models used in the initial draft were usable. The model chosen in the final draft was SSD v1. DL Workbench has not been discussed yet in the coursework. For this reason, the stats of the models from their README files in the Open Model Zoo were considered.

### 5.a) For SSD v1,
| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| GFLOPs            | 2.494         |
| MParams           | 6.807         |

### 5.b) For SSD v2,
| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| GFLOPs            | 3.775         |
| MParams           | 16.818        |

## 5.c) For SSD300,
| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| GFLOPs            | 62.815        |
| MParams           | 26.285        |

As can be easily observed, the v1 has less complexity and fewer model parameters. This is why SSD v1 was chosen.

**PS** -> It would have been really helpful if the OpenCV shipped with OpenVINO had "Trackers" which generally come OpenCV. Use of a tracker would have substantially sped up the whole process as I wouldn't have to infer on every frame and then process everytime after that. Inference would only happen when a person was lost or new person entered.
