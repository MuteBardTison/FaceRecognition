# Face Recognition

[![Build Status](https://travis-ci.org/MuteBardTison/FaceRecognition.svg?branch=master)](https://travis-ci.org/MuteBardTison/FaceRecognition)
[![release](http://github-release-version.herokuapp.com/github/MuteBardTison/FaceRecognition/release.svg?style=flat)](https://github.com/MuteBardTison/FaceRecognition/releases)

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

[Eigenfaces for Recognition](https://www.cs.ucsb.edu/~mturk/Papers/jcn.pdf). Turk, M. and Pentland, A. (1991). 

## License

  [MIT](https://github.com/MuteBardTison/FaceRecognition/blob/master/LICENSE) © Zihan Qi
