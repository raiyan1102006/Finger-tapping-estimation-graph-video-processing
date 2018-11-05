# Remote Diagnosis of Parkinson’s Disease from Finger-tapping Videos: A Graph Signal Processing Approach

This repository holds sample data files and the analysis code for the named project. The codes were developed by Raiyan Abdul Baten.

## What's in this Repository
The 'data' folder contains 4 sample videos of finger tapping. The **analysis.m** file analyzes one of the videos at a time and reports the extracted finger-tapping frequency. At the beginning of the script, the video file to load can be specified. The steps in the algorithm are commented in the code file. The detailed report and presentation slides that explain the algorithm can be found in this repository. 

## Installation and Usage

Please install the MATLAB package 'Image Processing Toolbox' from https://www.mathworks.com/products/image.html.

The **analysis.m** script can then be run from the MATLAB interface. 

## Implementation Summary
**Input:** A video file of finger-tapping.

**Output:** The finger-tapping frequency in Hz.

I have used a graph-based approach for extracting the finger-tapping frequency. The algorithm is detailed in [Project_Report.pdf](https://github.com/raiyan1102006/Finger_Tapping_Frequency_Estimation/blob/master/Project_Report.pdf).
