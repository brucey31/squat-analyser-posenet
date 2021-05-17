# Squat Analysis Posenet Project

![](https://github.com/brucey31/squat-analyser-posenet/blob/main/examples/Squat%20Analysis%20Video%20Demo.gif?raw=true)

## Purpose
The purpose of this project is to play with Posenet models to create something that can analyse the data coming out the end of it to do something useful. This project can use the position of each joint in your body (extracted using posenet) and then use the difference between the positions of these joints to assess whether the squats being performed by the person in the webcam has done a proper squat or not.

## How to use
* Follow the installation instructions below
* Start the Jupyter notebook and run all cells
* The webcam will show up after you run the second from bottom cell
* Step back from the webcam until it detects your shoulders, hips and ankles.
* The calibration process takes a couple of seconds and the text on the top left will count you in.
* Once calibration is complete, you can now start to squat.
* The border will flash green once a valid squat has been registered through the system and the squat count will be updated

## Installation instructions
* Clone the repo down to your local machine. The models are included, hence why its a little heavy.
* `pip install -r squat_analysis_requirements.txt`
* `jupyter notebook`

## Things to improve
* Instead of adding the model, it would be best to better use the code in the jupyter notebook that downloads it automatically
* I would love to be able to analyse squats for more than 1 person at a time
* This could also be made as a competition!!