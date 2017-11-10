/*
 FaceRecognition.cpp
 @author Zihan Qi
 2017/11/09
 */

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"
#include "boost/filesystem.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include <stdlib.h>

using namespace cv;
using namespace std;
using namespace boost;
using namespace boost::filesystem;

const int SampleNumPerImg = 10;

bool verify_folder(path& p) {
    if (!exists(p)) {
        cerr << "Folder " << p.c_str() << " does not exist" << endl;
        return false;
    }
    if (!(is_directory(p))) {
        cerr << p.c_str() << " is not a folder." << endl;
        return false;
    }
    return true;
}

struct not_digit {
    bool operator()(const char c) {
        return c != ' ' && !std::isdigit(c);
    }
};

int extract_int (string str) {
    not_digit not_a_digit;
    int n = 0;
    string::iterator end = remove_if(str.begin(), str.end(), not_a_digit);
    string all_numbers(str.begin(), end);
    stringstream ss(all_numbers);
    ss >> n;
    return n;
}


void load(path& p,vector<Mat>& images, vector<int>&labels){
    //using boost
    directory_iterator it(p);
    directory_iterator end_it;
    for (; it != end_it; it++) {
        path pf(it->path());
        if (is_directory(pf)) continue;
        string macOS_store("../yalefaces-centered/.DS_Store");
        if (pf.c_str() == macOS_store) continue;
        cout << "loading " << pf.c_str() << "..." << endl;
        //loading
        Mat itrain = imread(pf.c_str(),CV_LOAD_IMAGE_GRAYSCALE);
        if (!itrain.data){
            cerr<<"could not open"<<pf<<endl;
            exit(1);
        }
        images.push_back(itrain);
        labels.push_back(extract_int(pf.c_str()));
        cout << "label extracted: " << extract_int(pf.c_str()) << endl;
    }
}

// Normalizes a given image into a value range between 0 and 255.
static Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
        case 1:
            cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
            break;
        case 3:
            cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
            break;
        default:
            src.copyTo(dst);
            break;
    }
    return dst;
}

#ifdef READCSV

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}
#endif

int main(int argc, const char *argv[]) {
    // Check for valid command line arguments, print usage
    // if no arguments were given.
    if (argc < 3) {
        cout << "usage: " << argv[0] << " [training_folder] [test_folder]" << endl;
        exit(1);
    }
    //training folder
    path p_train (argv[1]);
    if (!verify_folder(p_train)) return -1;
    //test folder
    path p_test(argv[2]);
    if (!verify_folder(p_test)) return -1;
    //output folder
    string output_folder = ".";
    if (argc == 4) output_folder = string(argv[3]);
    
//csv read
#ifdef READCSV
    string fn_csv = string(argv[1]);
    // These vectors hold the images and corresponding labels.
    vector<Mat> images;
    vector<int> labels;
    // Read in the data. This can fail if no valid
    // input filename is given.
    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
#endif
    
    vector<Mat> images;
    vector<int> labels;
    vector<Mat> testimage;
    vector<int> testlabel;
    load(p_train, images,labels);
    cout<< "loaded "<<images.size()<< " training samples."<< endl;
    
    if(images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }
    
    int height = images[0].rows;
    //test instance- Yoda
    load(p_test, testimage,testlabel);
    Mat testSample=testimage[0];
    int testLabel=testlabel[0];
    
    //build the model
    Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
    model->train(images, labels);
    
    // test model
    int predictedLabel = model->predict(testSample);
    string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
    cout << result_message << endl;
    namedWindow("Yoda",WINDOW_AUTOSIZE);
    imshow("Yoda",testSample);
    imshow("ClosestImage",images.at(predictedLabel*SampleNumPerImg));
    waitKey(0);
    
    Mat eigenvalues = model->getMat("eigenvalues");
    Mat W = model->getMat("eigenvectors"); //display
    Mat mean = model->getMat("mean");
    if(argc == 3) {
        imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));
    } else {
        imwrite(format("%s/mean.png", output_folder.c_str()), norm_0_255(mean.reshape(1, images[0].rows)));
    }
    // Display or save the Eigenfaces:
    for (int i = 0; i < min(15, W.cols); i++) {
        string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
        cout << msg << endl;
        // get eigenvector #i
        Mat ev = W.col(i).clone();
        // Reshape to original size & normalize to [0...255] for imshow.
        Mat grayscale = norm_0_255(ev.reshape(1, height));
        // Show the image & apply a Jet colormap for better sensing.
        // Mat cgrayscale;
        //applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
        // Display or save:
        if(argc == 3) {
            imshow(format("eigenface_%d", i), grayscale);
        } else {
            imwrite(format("%s/eigenface_%d.png", output_folder.c_str(), i), norm_0_255(grayscale));
        }
    }
    
    for(int num_components = min(W.cols, 15); num_components < min(W.cols, 300); num_components+=15) {
        // slice the eigenvectors from the model
        Mat evs = Mat(W, Range::all(), Range(0, num_components));
        Mat projection = subspaceProject(evs, mean, images[0].reshape(1,1));
        Mat reconstruction = subspaceReconstruct(evs, mean, projection);
        // Normalize the result:
        reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));
        // Display or save:
        if(argc == 3) {
            imshow(format("eigenface_reconstruction_%d", num_components), reconstruction);
        } else {
            imwrite(format("%s/eigenface_reconstruction_%d.png", output_folder.c_str(), num_components), reconstruction);
        }
    }
    // Display if we are not writing to an output folder:
    if(argc == 3) {
        waitKey(0);
    }
    return 0;
}
