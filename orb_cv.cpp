#include<iostream>
#include<opencv2/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<chrono>

using namespace std;
using namespace cv;

int main(void){
    
    /*if(argc != 3)
    {
        
        cout << " usage: feature_extraction img1 img2"<< endl;
        return 1;
        
    }*/
    
    Mat img_1 = imread("/home/songzhimo/slam_training/ORB_featurePointExtract/test1.jpeg",1);
    Mat img_2 = imread("/home/songzhimo/slam_training/ORB_featurePointExtract/test2.jpeg",1);
    
    assert(img_1.data != nullptr && img_2.data != nullptr );
    //read the graph
    
     std::vector<KeyPoint> keypoints_1, keypoints_2;
     Mat description_1, description_2;
     Ptr<FeatureDetector> detector = ORB::create(); // the class of keypoint
     Ptr<DescriptorExtractor> descriptor = ORB::create(); // the class of descripe KeyPoint
     Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
     
     chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
     detector->detect(img_1, keypoints_1);
     detector->detect(img_2, keypoints_2);
     
     descriptor->compute(img_1,keypoints_1, description_1);
     descriptor->compute(img_2,keypoints_2, description_2);
     chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
     chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
     cout<<"time cost:"<< time_used.count()<<endl;

     Mat outimg1;
     drawKeypoints(img_1,keypoints_1,outimg1,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
     imshow("orb_features",outimg1);
     
     vector<DMatch> matches;
     chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
     matcher->match(description_1,description_2,matches);
     chrono::steady_clock::time_point t4 = chrono::steady_clock::now();
     chrono::duration<double> time_used_match = chrono::duration_cast<chrono::duration<double>>(t4-t3);
     cout<<"match time cost :"<< time_used_match.count()<<endl;

     auto min_max = minmax_element(matches.begin(),matches.end());
     double min_dist = min_max.first->distance;
     double max_dist = min_max.second->distance;
     
     printf("--max dist:%f\n",max_dist);
     printf("--min dist:%f\n",min_dist);

     vector<DMatch> good_match;
    for (int i = 0; i < description_1.rows; i++)
    {
        if (matches[i].distance <= max(2* min_dist,30.0))
        {
            good_match.push_back(matches[i]);
        }
    }

    Mat img_match;
    Mat img_goodmatch;
    drawMatches(img_1,keypoints_1,img_2,keypoints_2,matches,img_match);
    drawMatches(img_1,keypoints_1,img_2,keypoints_2,good_match,img_goodmatch);
    imshow("match",img_match);
    imshow("good_match",img_goodmatch);
    waitKey(0);

    return 0;
}
