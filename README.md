# Face Recognition

[![Build Status](https://travis-ci.org/MuteBardTison/FaceRecognition.svg?branch=master)](https://travis-ci.org/MuteBardTison/FaceRecognition)

This is an implementation of EigenFace Recognition Algorithm with OpenCV in C++.

Using approximately centered Yale-faces images and test Yoda image to find the closest image in the training dataset.

Using the boost filesystem library instead of the CSV file.

## Usage

```
cd <source directory>
mkdir build output
cd build
cmake ..
make
```
- without output folder

```
./FaceRecognition ../yalefaces-centered/ ../Test-Yoda/
```
- with output folder

```
./FaceRecognition ../yalefaces-centered/ ../Test-Yoda/ ../output
```

## Output

Demonstrating with iTerm2

![screenshot](https://user-images.githubusercontent.com/25029380/32647150-3cfd9df6-c5f0-11e7-939c-4d17654bba77.png)

## Reference

[Eigenfaces for Recognition](https://s3.amazonaws.com/academia.edu.documents/30894770/jcn.pdf?AWSAccessKeyId=AKIAIWOWYYGZ2Y53UL3A&Expires=1510301188&Signature=EOTg9mAN2ZdIbl4OqyYD1ZWeC4c%3D&response-content-disposition=inline%3B%20filename%3DEigenfaces_for_Recognition.pdf). Turk, M. and Pentland, A. (1991). 

## License

  [MIT](https://github.com/MuteBardTison/FaceRecognition/blob/master/LICENSE) © Zihan Qi
