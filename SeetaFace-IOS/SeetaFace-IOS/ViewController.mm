//
//  ViewController.m
//  SeetaFace-IOS
//
//  Created by zxs.zl on 2019/3/22.
//  Copyright © 2019 zxs.zl. All rights reserved.
//
#import <opencv2/opencv.hpp>
#import <opencv2/core/core_c.h>
#include "face_detection.h"
#import <Foundation/Foundation.h>

void faceDetect()
{
    NSString *nsImg_path = [[NSBundle mainBundle].bundlePath stringByAppendingPathComponent:@"0_1_1.jpg"];
    NSString *nsModel_path = [[NSBundle mainBundle].bundlePath stringByAppendingPathComponent:@"seeta_fd_frontal_v1.0.bin"];
    const char* img_path = [nsImg_path UTF8String];
    seeta::FaceDetection detector([nsModel_path UTF8String]);
    
    detector.SetMinFaceSize(40);
    detector.SetScoreThresh(2.f);
    detector.SetImagePyramidScaleFactor(0.8f);
    detector.SetWindowStep(4, 4);
    
    cv::Mat img = cv::imread(img_path, cv::IMREAD_UNCHANGED);
    cv::Mat img_gray;
    
    if (img.channels() != 1)
        cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    else
        img_gray = img;
    
    seeta::ImageData img_data;
    img_data.data = img_gray.data;
    img_data.width = img_gray.cols;
    img_data.height = img_gray.rows;
    img_data.num_channels = 1;
    
    long t0 = cv::getTickCount();
    std::vector<seeta::FaceInfo> faces = detector.Detect(img_data);
    long t1 = cv::getTickCount();
    double secs = (t1 - t0)/cv::getTickFrequency();
    
    std::cout << "Detections takes " << secs << " seconds " << std::endl;
    std::cout << "Image size (wxh): " << img_data.width << "x" << img_data.height << std::endl;
    
    cv::Rect face_rect;
    int32_t num_face = static_cast<int32_t>(faces.size());
    
    for (int32_t i = 0; i < num_face; i++) {
        face_rect.x = faces[i].bbox.x;
        face_rect.y = faces[i].bbox.y;
        face_rect.width = faces[i].bbox.width;
        face_rect.height = faces[i].bbox.height;
        
        cv::rectangle(img, face_rect, CV_RGB(0, 0, 255), 4, 8, 0);
    }
}

#include "face_alignment.h"
void face_alignment()
{
    NSString *nsModel_path = [[NSBundle mainBundle].bundlePath stringByAppendingPathComponent:@"seeta_fd_frontal_v1.0.bin"];
    seeta::FaceDetection detector([nsModel_path UTF8String]);
    detector.SetMinFaceSize(40);
    detector.SetScoreThresh(2.f);
    detector.SetImagePyramidScaleFactor(0.8f);
    detector.SetWindowStep(4, 4);
    
    // Initialize face alignment model
    nsModel_path = [[NSBundle mainBundle].bundlePath stringByAppendingPathComponent:@"seeta_fa_v1.1.bin"];
    seeta::FaceAlignment point_detector([nsModel_path UTF8String]);
    
    //load image
    NSString *nsImg_path = [[NSBundle mainBundle].bundlePath stringByAppendingPathComponent:@"image_0001.png"];
    
    cv::Mat img_color = cv::imread([nsImg_path UTF8String], cv::IMREAD_ANYCOLOR);
    cv::Mat img_grayscale;
    if (img_color.channels() != 1)
        cv::cvtColor(img_color, img_grayscale, cv::COLOR_BGR2GRAY);
    else
        img_grayscale = img_color;
    
    int pts_num = 5;
    int im_width = img_grayscale.cols;
    int im_height = img_grayscale.rows;
    unsigned char* data = new unsigned char[im_width * im_height];
    unsigned char* data_ptr = data;
    unsigned char* image_data_ptr = (unsigned char*)img_grayscale.data;
    int h = 0;
    for (h = 0; h < im_height; h++) {
        memcpy(data_ptr, image_data_ptr, im_width);
        data_ptr += im_width;
        image_data_ptr += img_grayscale.step;
    }
    
    seeta::ImageData image_data;
    image_data.data = data;
    image_data.width = im_width;
    image_data.height = im_height;
    image_data.num_channels = 1;
    
    // Detect faces
    std::vector<seeta::FaceInfo> faces = detector.Detect(image_data);
    int32_t face_num = static_cast<int32_t>(faces.size());
    
    if (face_num == 0)
    {
        delete[]data;
        return;
    }
    
    // Detect 5 facial landmarks
    seeta::FacialLandmark points[5];
    point_detector.PointDetectLandmarks(image_data, faces[0], points);
    
    // Visualize the results
    cv::rectangle(img_color, cvPoint(faces[0].bbox.x, faces[0].bbox.y), cvPoint(faces[0].bbox.x + faces[0].bbox.width - 1, faces[0].bbox.y + faces[0].bbox.height - 1), CV_RGB(255, 0, 0));
    for (int i = 0; i<pts_num; i++)
    {
        cv::circle(img_color, cvPoint(points[i].x, points[i].y), 2, CV_RGB(0, 255, 0));
    }
    cv::imwrite("/你自己的路径/result.jpg", img_color);
    // Release memory
    delete[]data;
}

#include "face_identification.h"

void face_verification()
{
    NSString *nsModel_path = [[NSBundle mainBundle].bundlePath stringByAppendingPathComponent:@"seeta_fd_frontal_v1.0.bin"];
    seeta::FaceDetection detector([nsModel_path UTF8String]);
    detector.SetMinFaceSize(40);
    detector.SetScoreThresh(2.f);
    detector.SetImagePyramidScaleFactor(0.8f);
    detector.SetWindowStep(4, 4);
    
    // Initialize face alignment model
    nsModel_path = [[NSBundle mainBundle].bundlePath stringByAppendingPathComponent:@"seeta_fa_v1.1.bin"];
    seeta::FaceAlignment point_detector([nsModel_path UTF8String]);
    
    // Initialize face Identification model
    nsModel_path = [[NSBundle mainBundle].bundlePath stringByAppendingPathComponent:@"seeta_fr_v1.0.bin"];
    seeta::FaceIdentification face_recognizer([nsModel_path UTF8String]);
    std::string test_dir = [[[NSBundle mainBundle].bundlePath stringByAppendingString:@"/test_face_recognizer/"] UTF8String];
    
    //load image
    cv::Mat gallery_img_color = cv::imread(test_dir + "images/compare_im/Aaron_Peirsol_0001.jpg", cv::IMREAD_COLOR);
    cv::Mat gallery_img_gray;
    cv::cvtColor(gallery_img_color, gallery_img_gray, cv::COLOR_BGR2GRAY);
    
    cv::Mat probe_img_color = cv::imread(test_dir + "images/compare_im/Aaron_Peirsol_0004.jpg", cv::IMREAD_COLOR);
    cv::Mat probe_img_gray;
    cv::cvtColor(probe_img_color, probe_img_gray, cv::COLOR_BGR2GRAY);
    
    seeta::ImageData gallery_img_data_color(gallery_img_color.cols, gallery_img_color.rows, gallery_img_color.channels());
    gallery_img_data_color.data = gallery_img_color.data;
    
    seeta::ImageData gallery_img_data_gray(gallery_img_gray.cols, gallery_img_gray.rows, gallery_img_gray.channels());
    gallery_img_data_gray.data = gallery_img_gray.data;
    
    seeta::ImageData probe_img_data_color(probe_img_color.cols, probe_img_color.rows, probe_img_color.channels());
    probe_img_data_color.data = probe_img_color.data;
    
    seeta::ImageData probe_img_data_gray(probe_img_gray.cols, probe_img_gray.rows, probe_img_gray.channels());
    probe_img_data_gray.data = probe_img_gray.data;
    
    // Detect faces
    std::vector<seeta::FaceInfo> gallery_faces = detector.Detect(gallery_img_data_gray);
    int32_t gallery_face_num = static_cast<int32_t>(gallery_faces.size());
    
    std::vector<seeta::FaceInfo> probe_faces = detector.Detect(probe_img_data_gray);
    int32_t probe_face_num = static_cast<int32_t>(probe_faces.size());
    
    if (gallery_face_num == 0 || probe_face_num==0)
    {
        std::cout << "Faces are not detected.";
        return;
    }
    
    // Detect 5 facial landmarks
    seeta::FacialLandmark gallery_points[5];
    point_detector.PointDetectLandmarks(gallery_img_data_gray, gallery_faces[0], gallery_points);
    
    seeta::FacialLandmark probe_points[5];
    point_detector.PointDetectLandmarks(probe_img_data_gray, probe_faces[0], probe_points);
    
    for (int i = 0; i<5; i++)
    {
        cv::circle(gallery_img_color, cv::Point(gallery_points[i].x, gallery_points[i].y), 2,
                   CV_RGB(0, 255, 0));
        cv::circle(probe_img_color, cv::Point(probe_points[i].x, probe_points[i].y), 2,
                   CV_RGB(0, 255, 0));
    }
    cv::imwrite("/你自己的路径/gallery_point_result.jpg", gallery_img_color);
    cv::imwrite("/你自己的路径/probe_point_result.jpg", probe_img_color);
    
    // Extract face identity feature
    float gallery_fea[2048];
    float probe_fea[2048];
    face_recognizer.ExtractFeatureWithCrop(gallery_img_data_color, gallery_points, gallery_fea);
    face_recognizer.ExtractFeatureWithCrop(probe_img_data_color, probe_points, probe_fea);
    
    // Caculate similarity of two faces
    float sim = face_recognizer.CalcSimilarity(gallery_fea, probe_fea);
    std::cout << sim <<std::endl;
}


#include "face_identification.h"
#include <iostream>
#include <fstream>

#define TEST(major, minor) major##_##minor##_Tester()
#define EXPECT_NE(a, b) if ((a) == (b)) std::cout << "ERROR: "
#define EXPECT_EQ(a, b) if ((a) != (b)) std::cout << "ERROR: "


void TEST(FaceRecognizerTest, CropFace)
{
    NSString *nsModel_path = [[NSBundle mainBundle].bundlePath stringByAppendingPathComponent:@"seeta_fr_v1.0.bin"];
    seeta::FaceIdentification face_recognizer([nsModel_path UTF8String]);
    std::string test_dir = [[[NSBundle mainBundle].bundlePath stringByAppendingString:@"/test_face_recognizer/"] UTF8String];
    /* data initialize */
    std::ifstream ifs;
    std::string img_name;
    seeta::FacialLandmark pt5[5];
    ifs.open(test_dir + "test_file_list.txt", std::ifstream::in);
    clock_t start, count = 0;
    int img_num = 0;
    while (ifs >> img_name) {
        img_num ++ ;
        // read image
        cv::Mat src_img = cv::imread(test_dir + img_name, 1);
        EXPECT_NE(src_img.data, nullptr) << "Load image error!";
        
        // ImageData store data of an image without memory alignment.
        seeta::ImageData src_img_data(src_img.cols, src_img.rows, src_img.channels());
        src_img_data.data = src_img.data;
        
        // 5 located landmark points (left eye, right eye, nose, left and right
        // corner of mouse).
        for (int i = 0; i < 5; ++ i) {
            ifs >> pt5[i].x >> pt5[i].y;
        }
        
        // Create a image to store crop face.
        cv::Mat dst_img(face_recognizer.crop_height(),
                        face_recognizer.crop_width(),
                        CV_8UC(face_recognizer.crop_channels()));
        seeta::ImageData dst_img_data(dst_img.cols, dst_img.rows, dst_img.channels());
        dst_img_data.data = dst_img.data;
        /* Crop Face */
        start = clock();
        face_recognizer.CropFace(src_img_data, pt5, dst_img_data);
        count += clock() - start;
        // Show crop face
        //    cv::imshow("Crop Face", dst_img);
        //    cv::waitKey(0);
        //    cv::destroyWindow("Crop Face");
    }
    ifs.close();
    std::cout << "Test successful! \nAverage crop face time: "
    << 1000.0 * count / CLOCKS_PER_SEC / img_num << "ms" << std::endl;
}

void TEST(FaceRecognizerTest, ExtractFeature) {
    NSString *nsModel_path = [[NSBundle mainBundle].bundlePath stringByAppendingPathComponent:@"seeta_fr_v1.0.bin"];
    seeta::FaceIdentification face_recognizer([nsModel_path UTF8String]);
    std::string test_dir = [[[NSBundle mainBundle].bundlePath stringByAppendingString:@"/test_face_recognizer/"] UTF8String];
    
    int feat_size = face_recognizer.feature_size();
    EXPECT_EQ(feat_size, 2048);
    
    FILE* feat_file = NULL;
    
    // Load features extract from caffe
    feat_file = fopen((test_dir + "feats.dat").c_str(), "rb");
    int n, c, h, w;
    EXPECT_EQ(fread(&n, sizeof(int), 1, feat_file), (unsigned int)1);
    EXPECT_EQ(fread(&c, sizeof(int), 1, feat_file), (unsigned int)1);
    EXPECT_EQ(fread(&h, sizeof(int), 1, feat_file), (unsigned int)1);
    EXPECT_EQ(fread(&w, sizeof(int), 1, feat_file), (unsigned int)1);
    float* feat_caffe = new float[n * c * h * w];
    float* feat_sdk = new float[n * c * h * w];
    EXPECT_EQ(fread(feat_caffe, sizeof(float), n * c * h * w, feat_file),
              n * c * h * w);
    EXPECT_EQ(feat_size, c * h * w);
    
    //    int cnt = 0;
    
    /* Data initialize */
    std::ifstream ifs(test_dir + "crop_file_list.txt");
    std::string img_name;
    
    clock_t start, count = 0;
    int img_num = 0, lb;
    double average_sim = 0.0;
    while (ifs >> img_name >> lb) {
        // read image
        cv::Mat src_img = cv::imread(test_dir + img_name, 1);
        EXPECT_NE(src_img.data, nullptr) << "Load image error!";
        cv::resize(src_img, src_img, cv::Size(face_recognizer.crop_height(),
                                              face_recognizer.crop_width()));
        
        // ImageData store data of an image without memory alignment.
        seeta::ImageData src_img_data(src_img.cols, src_img.rows, src_img.channels());
        src_img_data.data = src_img.data;
        
        /* Extract feature */
        start = clock();
        face_recognizer.ExtractFeature(src_img_data,
                                       feat_sdk + img_num * feat_size);
        count += clock() - start;
        
        /* Caculate similarity*/
        float* feat1 = feat_caffe + img_num * feat_size;
        float* feat2 = feat_sdk + img_num * feat_size;
        float sim = face_recognizer.CalcSimilarity(feat1, feat2);
        average_sim += sim;
        img_num ++ ;
    }
    ifs.close();
    average_sim /= img_num;
    if (1.0 - average_sim >  0.01) {
        std::cout<< "average similarity: " << average_sim << std::endl;
    }
    else {
        std::cout << "Test successful!\nAverage extract feature time: "
        << 1000.0 * count / CLOCKS_PER_SEC / img_num << "ms" << std::endl;
    }
    delete []feat_caffe;
    delete []feat_sdk;
}

void TEST(FaceRecognizerTest, ExtractFeatureWithCrop) {
    NSString *nsModel_path = [[NSBundle mainBundle].bundlePath stringByAppendingPathComponent:@"seeta_fr_v1.0.bin"];
    seeta::FaceIdentification face_recognizer([nsModel_path UTF8String]);
    std::string test_dir = [[[NSBundle mainBundle].bundlePath stringByAppendingString:@"/test_face_recognizer/"] UTF8String];
    
    int feat_size = face_recognizer.feature_size();
    EXPECT_EQ(feat_size, 2048);
    
    FILE* feat_file = NULL;
    
    // Load features extract from caffe
    feat_file = fopen((test_dir + "feats.dat").c_str(), "rb");
    int n, c, h, w;
    EXPECT_EQ(fread(&n, sizeof(int), 1, feat_file), (unsigned int)1);
    EXPECT_EQ(fread(&c, sizeof(int), 1, feat_file), (unsigned int)1);
    EXPECT_EQ(fread(&h, sizeof(int), 1, feat_file), (unsigned int)1);
    EXPECT_EQ(fread(&w, sizeof(int), 1, feat_file), (unsigned int)1);
    float* feat_caffe = new float[n * c * h * w];
    float* feat_sdk = new float[n * c * h * w];
    EXPECT_EQ(fread(feat_caffe, sizeof(float), n * c * h * w, feat_file),
              n * c * h * w);
    EXPECT_EQ(feat_size, c * h * w);
    
    //    int cnt = 0;
    
    /* Data initialize */
    std::ifstream ifs(test_dir + "test_file_list.txt");
    std::string img_name;
    seeta::FacialLandmark pt5[5];
    
    clock_t start, count = 0;
    int img_num = 0;
    double average_sim = 0.0;
    while (ifs >> img_name) {
        // read image
        cv::Mat src_img = cv::imread(test_dir + img_name, 1);
        EXPECT_NE(src_img.data, nullptr) << "Load image error!";
        
        // ImageData store data of an image without memory alignment.
        seeta::ImageData src_img_data(src_img.cols, src_img.rows, src_img.channels());
        src_img_data.data = src_img.data;
        
        // 5 located landmark points (left eye, right eye, nose, left and right
        // corner of mouse).
        for (int i = 0; i < 5; ++ i) {
            ifs >> pt5[i].x >> pt5[i].y;
        }
        
        /* Extract feature: ExtractFeatureWithCrop */
        start = clock();
        face_recognizer.ExtractFeatureWithCrop(src_img_data, pt5,
                                               feat_sdk + img_num * feat_size);
        count += clock() - start;
        
        /* Caculate similarity*/
        float* feat1 = feat_caffe + img_num * feat_size;
        float* feat2 = feat_sdk + img_num * feat_size;
        float sim = face_recognizer.CalcSimilarity(feat1, feat2);
        average_sim += sim;
        img_num ++ ;
    }
    ifs.close();
    average_sim /= img_num;
    if (1.0 - average_sim >  0.02) {
        std::cout<< "average similarity: " << average_sim << std::endl;
    }
    else {
        std::cout << "Test successful!\nAverage extract feature time: "
        << 1000.0 * count / CLOCKS_PER_SEC / img_num << "ms" << std::endl;
    }
    delete []feat_caffe;
    delete []feat_sdk;
}

void face_recognizer()
{
    TEST(FaceRecognizerTest, CropFace);
    TEST(FaceRecognizerTest, ExtractFeature);
    TEST(FaceRecognizerTest, ExtractFeatureWithCrop);
}


#import "ViewController.h"


@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    //人脸检测
    faceDetect();
    
    //特征点定位 眼睛 鼻子 嘴
    //face_alignment();
    
    //人脸特征核实 验证 “我是我”
    //face_verification();
    
    //人脸特征比对 比对 “我是谁”
    //face_recognizer();
}

@end
