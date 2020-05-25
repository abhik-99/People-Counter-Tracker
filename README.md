# Project Write-Up

### Project People Counter App
**BY** - Abhik Banerjee
**Contact** - abhik@abhikbanerjee.com, abhik.banerjee.1999@gmail.com
**Model Used** - Single-shot detector V1 trained on CoCo Dataset
**Link to Test Video** - https://youtu.be/aMronQi4H4I

#### Please go to Submission > Project Write-up for actual Submission to this project.

## Assess Model Use Cases

From the very Video that was given for usage, I can think of one immediate use case:-
1. Tracking the behaviour of a person in a given area of interest. The People in the video seemed to enter the frame, read from a piece of paper and then leave from one specific side. The Model can be used to predict if the person takes the wrong step - this can be detected via tracking the position of the centroid.
2. The model can be used to assess the time a person spends in the frame. 

These two points can also be commonly observed in *ATMs* were at a given time only 1 person should be present in the frame and they should not be present more than a specific alloted time. 

Centroid tracking was used in the project. People who were actively being tracked were marked with a green centroid while those who have exited the frame were denoted by a red centroid at the last point they were spotted. This specific feature can be used for trajectory as well as most favoured point of exit prediction.

# Deploy a People Counter App at the Edge

| Details            |              |
|-----------------------|---------------|
| Programming Language: |  Python 3.5 or 3.6 |

![people-counter-python](./images/people-counter-image.png)

## What it Does

The people counter application will demonstrate how to create a smart video IoT solution using Intel® hardware and software tools. The app will detect people in a designated area, providing the number of people in the frame, average duration of people in frame, and total count.

## How it Works

The counter will use the Inference Engine included in the Intel® Distribution of OpenVINO™ Toolkit. The model used should be able to identify people in a video frame. The app should count the number of people in the current frame, the duration that a person is in the frame (time elapsed between entering and exiting a frame) and the total count of people. It then sends the data to a local web server using the Paho MQTT Python package.



### Software

*   Intel® Distribution of OpenVINO™ toolkit 2019 R3 release
*   Node v6.17.1
*   Npm v3.10.10
*   CMake
*   MQTT Mosca server
  
        
## Setup

### Install Intel® Distribution of OpenVINO™ toolkit

Utilize the classroom workspace, or refer to the relevant instructions for your operating system for this step.

- [Linux/Ubuntu](./linux-setup.md)
- [Mac](./mac-setup.md)
- [Windows](./windows-setup.md)

### Install Nodejs and its dependencies

Utilize the classroom workspace, or refer to the relevant instructions for your operating system for this step.

- [Linux/Ubuntu](./linux-setup.md)
- [Mac](./mac-setup.md)
- [Windows](./windows-setup.md)


