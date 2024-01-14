<h1 align=center>🚗: CARCOU

![](https://img.shields.io/badge/Python-3.9-blue) ![](https://img.shields.io/badge/opencv-4.8.1.78-blue) ![](https://img.shields.io/badge/Contributions-Welcome-brightgreen) ![](https://img.shields.io/badge/LICENSE-MIT-red)</h1>

A simple application that counts the number of cars crossing a red line.

## Overview
<p align = left>Cars are detected using YOLOv8 nano version and tracked using <a href='https://github.com/abewley/sort'>SORT</a> algorithm. When a car with a unique tracking id crosses the red line, the line color momentarily changes to green and the count value is increased accordingly.</p>

## Hyperparameters

If you want to reuse this code for your own projects then these are the values that you need to change.  
* Enter a class name/s you want to detect.
```python
CLASSES = ['car', 'bus']
```
* Define limits of a detection line using (x1, y1) and (x2, y2) coordinate system.
```python
LIMITS = [915, 1575, 2800, 1575]
```
Additionally, you will need to create a new **mask.png** for your project.

## Demo
<p align = left>Here is a quick demo of the application.</p>
<p align="center">
  <img src = "https://github.com/kotiyalanurag/CARCOU/assets/44325770/76b04cde-6922-424e-9ec5-3a909700041e">
</p>
<p align = left>The cars are only detected inside a specified region due to an image mask which was created using <a href='https://www.canva.com/'>canva.</a></p>