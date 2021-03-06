# EyeTracking-to-find-the-AreaofInterest

This project is about tracking the user eye movement and finding the particular **Area of Interest** from the whole screen. Its application starts from using it give more personalised advertisements based on which region our eye is looking upon, It can be used to detect our concentration level by monitoring our eye movement etc


You might find the code under the name "GazeTracking_main". before executing the code you might need some prerequisite which can be found in the last part of this file

The program uses Haarcascade classifier to detect the eye regoin and then it uses image Threasholding and Processing to filter out the iris part which is crucial for GazeTracking.

The Gaze tracking works live and tracks the instataneous region of interest of the user and records it

Finally it produces a BarChart to visualise the recorded data. Here you can observe which are the regions of ht monitor the user is more interested and which are the places of least interest

Model output is available under the name "Model Stastical data"







**prerequisite:** Python 3.5(preferred), Dlib, Haarcascade classifier, open-cv, numpy


