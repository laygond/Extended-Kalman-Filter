# Extended Kalman Filter

In this project a kalman filter is used to estimate the state of a moving object from the vehicle's perspective. This moving object of interest can be a cyclist, pedestrian, etc and tracking its position and velocity is measured with noisy lidar and radar measurements. This repo uses [Udacity's CarND-Extended-Kalman-Filter-Project repo](https://github.com/udacity/CarND-Extended-Kalman-Filter-Project) as a base template and guide. 

[//]: # (List of Images used in this README.md)
[image1]: ./README_images/visualization.gif "Visualization"
[image2]: ./README_images/traffic_sign_catalog.png "Catalog"
[image3]: ./README_images/train_set_dist.png "Training Set Distribution"
[image4]: ./README_images/architecture.png "Model Architecture"
[image5]: ./README_images/NNparam.png "Model Parameters"
[image6]: ./README_images/traffic_signs.png "Traffic Signs"
[image7]: ./README_images/stopinspect.png "Stop Sign Inspect"

![alt text][image1]


## Directory Structure
```
.Traffic-Sign-Classifier
├── demo.ipynb                   # Main file
build # not included (follow basic build instructions under dependency)
├── .gitignore                   # git file to prevent unnecessary files from being uploaded
├── README_images                # Images used by README.md
│   └── ...
├── README.md
├── net_weights                  # the model's kernel weights are stored here
│   └── ...
└── dataset
    ├── German_Traffic_Sign_Dataset
    │   ├── signnames.csv        # name and ID of each traffic sign in the dataset
    │   ├── my_test_images       # images found by me on the web for testing (ADD YOUR OWN HERE)
    │   │   └── ...
    │   ├── test.p               # second collection of test images for testing and it comes
    │   ├── train.p              # \\in the same format as training and validation, i.e, resized
    │   └── valid.p              # \\images stored as pickle files 
    └── US_Traffic_Sign_Dataset
```



## Steps to Launch Simulation
#### Dependencies (Just do Once)
Term 2 Simulator can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases) all other dependencies can be installed by running either 'install-linux.sh' or 'install-mac.sh' from a terminal. If you have windows use the [Ubuntu Bash Shell](https://www.howtogeek.com/249966/how-to-install-and-use-the-linux-bash-shell-on-windows-10/) and run 'install-linux.sh'.

1. Open a Terminal
2. Clone this repo: 'git clone https://github.com/laygond/Extended-Kalman-Filter'
3. Change directory: 'cd Extended-Kalman-Filter' 
4. Make shell file executable: 'sudo chmod +x ./install-linux.sh' (or 'install-mac.sh')
5. Run the file: 'sudo ./install-linux.sh' (or 'install-mac.sh')
 
 This file will install
* cmake >= 3.5
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
* gcc/g++ >= 5.4
* [uWebSocketIO](https://github.com/uWebSockets/uWebSockets)   #This allows for communication bewtween c++ code and simulator 

if you have problems installing these dependencies go [here](https://github.com/udacity/CarND-Extended-Kalman-Filter-Project) under dependencies and give those instructions a try.

#### Basic Build Instructions
Look at Directory Structure for reference

1. Make a build directory inside the Extended-Kalman-Filter repo: `mkdir build && cd build` 
2. Compile cmake file: `cmake .. && make` 
3. Run it: `./ExtendedKF ` 

If you modify your code, delete the build folder and follow the "Basic Build Instructions" again


## Output
// Specify coordinate system otherwise (sin and cos flip)
// Passing the project requires obtaining RMSE values that are lower than the tolerance outlined in the project rubric. 
// Please see the uWebSocketIO Starter Guide page in the classroom within the EKF Project lesson for the required version and installation scripts.



## Demo File
#### Overview
The demo file has the following sections:

- Load the data set (see below for links to the project data set)
- Explore and visualize the data set
- Model architecture Design
- Train and Validate Model
- Test Model
- Inpection & Analysis of Model

#### Dataset
The demo file makes use of the German traffic sign dataset to show results. However, once you have run and understood the `demo.ipynb`, feel free to try your own dataset by changing the input directories from 'Load The Data' section in the demo file.

The dataset provided for training and validation are pickle files containing RESIZED VERSIONS (32 by 32) of the [original dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). There exist two test sets for evaluation, one as a pickle file with the same instructions as validation and training set, and the other is `my_test_images` with images I collected from the web.

The following is a catalog of the signs in the dataset as labeled in `signnames.csv`:

![alt text][image2]

The dataset contains 43 classes labeled from [0-42]. The image shape is (32, 32, 3) for all pickle files while `my_test_images` has images of random sizes which are later preprocessed to be resized to (32, 32, 3). <b>ADD YOUR OWN GERMAN TRAFFIC SIGNS FOR TESTING IN  `my_test_images`.</b>

| Set | Samples |
| :---  | :--------: |
| Training Set   | 34799  |
| Validation Set |  4410  |
| Test Set       | 12630  |
| My Test Set    |     6  |

The distribution of the training set is:
![alt text][image3]


#### Model Architecture
The model of the neural network has a LeNet architecture with dropout added in the fully connected layers to improve accuracy. The LeNet architecture requires the input image to be 32x32 so every image goes through preprocessing for rescaling and normalization. Normalization is done by substracting from every image the mean and dividing it by the standard deviation. Both mean and std are from the training set. The following image shows the architecture and its specifications.

![alt text][image4]


###### Note: 

- `d = 075` in this case means you keep 75% of the fully connected layer
- Normalization helps to find faster better weights during training 
- The dropout added to the LeNet architecture increased the training accuracy by 4% than without.
- The optimizer used was Adam
- The Networks parameters were  the following:

![alt text][image5]


## Analysis of Model on Test Images

Here are five random German traffic signs that I found on the web:

![alt text][image6]

The model's top 5 softmax predictions on each of these new traffic signs were:

<b>Stop</b> 

| Prob: | Top 5 Predictions |
| ---:  | :-------- |
| 100.0% | Priority road |
| 0.0%   | End of no passing by vehicles over 3.5 metric tons |
| 0.0%   | No passing for vehicles over 3.5 metric tons |
| 0.0%   | End of no passing |
| 0.0%   | Right-of-way at the next intersection |


<b>Children crossing</b>

| Prob: | Top 5 Predictions |
| ---:  | :-------- |
| 99.64%  | Children crossing |
| 0.3%    | Pedestrians |
| 0.03%   | Right-of-way at the next intersection |
| 0.02%   | Road narrows on the right |
| 0.01%   | Dangerous curve to the right |


<b>Go straight or right</b>

| Prob: | Top 5 Predictions |
| ---:  | :-------- |
| 100.0% | Go straight or right |
| 0.0%   | General caution |
| 0.0%   | Keep right |
| 0.0%   | Roundabout mandatory |
| 0.0%   | Turn left ahead |


<b>Speed limit (30km/h)</b>

| Prob: | Top 5 Predictions |
| ---:  | :-------- |
| 98.78% | Speed limit (30km/h) |
| 0.86%  | Speed limit (20km/h) |
| 0.36%  | Speed limit (70km/h) |
| 0.0%   | Speed limit (50km/h) |
| 0.0%   | Speed limit (80km/h) |


<b>Roundabout mandatory</b>

| Prob: | Top 5 Predictions |
| ---:  | :-------- |
| 99.96% | Roundabout mandatory | 
| 0.04%  | Turn right ahead |
| 0.0%   | Ahead only |
| 0.0%   | Go straight or left |
| 0.0%   | Turn left ahead |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the original test set of 100.0 %

The `STOP` traffic sign failed in the prediction with `Priority Road` so let's inspect the its feature maps in for LeNet's first convolutional, first max pool, and second convolutional layers.

![alt text][image7]


## Drawbacks and improvements
The architecture is very basic, although we were able to reach:

| Set   | Accuracy |
| :---  | --------: |
| Validation Set |   95.4% |
| Test Set       | 100.0%  |
| My Test Set    |     80%  |

Possible improvements would be to make the architecture deeper to reach higher accuracy, evaluate how well the model performs if trained on other countries\` traffic signs, add detection to classify multiple traffic signs per image.   





Self-Driving Car Engineer Nanodegree Program

// Specify coordinate system otherwise (sin and cos flip)

In this project you will utilize a kalman filter to estimate the state of a moving object of interest with noisy lidar and radar measurements. Passing the project requires obtaining RMSE values that are lower than the tolerance outlined in the project rubric. 

This project involves the Term 2 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases)

This repository includes two files that can be used to set up and install [uWebSocketIO](https://github.com/uWebSockets/uWebSockets) for either Linux or Mac systems. For windows you can use either Docker, VMware, or even [Windows 10 Bash on Ubuntu](https://www.howtogeek.com/249966/how-to-install-and-use-the-linux-bash-shell-on-windows-10/) to install uWebSocketIO. Please see the uWebSocketIO Starter Guide page in the classroom within the EKF Project lesson for the required version and installation scripts.

Once the install for uWebSocketIO is complete, the main program can be built and run by doing the following from the project top directory.

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./ExtendedKF


'main.cpp' uses uWebSocketIO in communicating with the simulator.


Simulator (Client) ->  The c++ program (Server)
INPUT: (values provided by the Client)
["sensor_measurement"] => the measurement that the simulator observed (either lidar or radar)

OUTPUT: values provided by the c++ program to the simulator
["estimate_x"] <= kalman filter estimated position x
["estimate_y"] <= kalman filter estimated position y
["rmse_x"]
["rmse_y"]
["rmse_vx"]
["rmse_vy"]

---


## Dataset
#### in code

Explanation of the Data File
The github repo contains one data file:

obj_pose-laser-radar-synthetic-input.txt
Here is a screenshot of the first data file:

The simulator will be using this data file, and feed main.cpp values from it one line at a time.


Screenshot of Data File

Each row represents a sensor measurement where the first column tells you if the measurement comes from radar (R) or lidar (L).

For a row containing radar data, the columns are: sensor_type, rho_measured, phi_measured, rhodot_measured, timestamp, x_groundtruth, y_groundtruth, vx_groundtruth, vy_groundtruth, yaw_groundtruth, yawrate_groundtruth.

For a row containing lidar data, the columns are: sensor_type, x_measured, y_measured, timestamp, x_groundtruth, y_groundtruth, vx_groundtruth, vy_groundtruth, yaw_groundtruth, yawrate_groundtruth.

Whereas radar has three measurements (rho, phi, rhodot), lidar has two measurements (x, y).

You will use the measurement values and timestamp in your Kalman filter algorithm. Groundtruth, which represents the actual path the bicycle took, is for calculating root mean squared error.

You do not need to worry about yaw and yaw rate ground truth values.

Reading in the Data
We have provided code that will read in and parse the data files for you. This code is in the main.cpp file. The main.cpp file creates instances of a MeasurementPackage.

If you look inside 'main.cpp', you will see code like:

MeasurementPackage meas_package;
meas_package.sensor_type_ = MeasurementPackage::LASER;
meas_package.raw_measurements_ = VectorXd(2);
meas_package.raw_measurements_ << px, py;
meas_package.timestamp_ = timestamp;
and

vector<VectorXd> ground_truth;
VectorXd gt_values(4);
gt_values(0) = x_gt;
gt_values(1) = y_gt; 
gt_values(2) = vx_gt;
gt_values(3) = vy_gt;
ground_truth.push_back(gt_values);
The code reads in the data file line by line. The measurement data for each line gets pushed onto a measurement_pack_list. The ground truth [p_x, p_y, v_x, v_y][p 
x
​	 ,p 
y
​	 ,v 
x
​	 ,v 
y
​	 ] for each line in the data file gets pushed ontoground_truthso RMSE can be calculated later from tools.cpp.

#### Generating Additional Data

This is optional!

If you'd like to generate your own radar and lidar data, see the
[utilities repo](https://github.com/udacity/CarND-Mercedes-SF-Utilities) for
Matlab scripts that can generate additional data.

## Project Instructions and Rubric

## Hints and Tips!

* You don't have to follow this directory structure, but if you do, your work
  will span all of the .cpp files here. Keep an eye out for TODOs.
* Students have reported rapid expansion of log files when using the term 2 simulator.  This appears to be associated with not being connected to uWebSockets.  If this does occur,  please make sure you are conneted to uWebSockets. The following workaround may also be effective at preventing large log files.

    + create an empty log file
    + remove write permissions so that the simulator can't write to log
 * Please note that the ```Eigen``` library does not initialize ```VectorXd``` or ```MatrixXd``` objects with zeros upon creation.

## Call for IDE Profiles Pull Requests

Help your fellow students!

We decided to create Makefiles with cmake to keep this project as platform
agnostic as possible. Similarly, we omitted IDE profiles in order to ensure
that students don't feel pressured to use one IDE or another.

However! We'd love to help people get up and running with their IDEs of choice.
If you've created a profile for an IDE that you think other students would
appreciate, we'd love to have you add the requisite profile files and
instructions to ide_profiles/. For example if you wanted to add a VS Code
profile, you'd add:

* /ide_profiles/vscode/.vscode
* /ide_profiles/vscode/README.md

The README should explain what the profile does, how to take advantage of it,
and how to install it.

Regardless of the IDE used, every submitted project must
still be compilable with cmake and make.
