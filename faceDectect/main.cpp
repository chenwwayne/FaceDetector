#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\objdetect\objdetect.hpp>
#include<stdio.h>
#include<vector>

using namespace std;
using namespace cv;

CascadeClassifier faceCascade ;//创建一个关于人脸的级联分类器
CascadeClassifier eyeCascade;//创建一个关于眼睛的级联分类器
CascadeClassifier mouthCascade;//创建一个关于嘴巴的级联分类器
CascadeClassifier noseCascade;//创建一个关于鼻子的级联分类器

String faceCascadeName = "haarcascade_frontalface_alt.xml";//人脸的训练数据
String eyeCascadeName = "haarcascade_eye_tree_eyeglasses.xml";//眼睛的训练数据
String mouthCascadeName = "haarcascade_mcs_mouth.xml";//嘴巴的训练数据
String noseCascadeName = "haarcascade_mcs_nose.xml";//鼻子的训练数据
String windowName = "Face Dectect by chanweiwei";//窗口名称

void detectAndShow(Mat pic)
{
	vector<Rect> faces;
	Mat pic_gray;
	double timeCount;

	//RGB->灰度图
	cvtColor(pic, pic_gray, CV_RGB2GRAY);
	//直方图均衡化，可以将比较淡的图像变换为比较深的图像（即增强图像的亮度及对比度）。
	//数学原理:一个分布（输入的亮度直方图）被映射到另一个分布（一个更宽，理想统一的亮度值分布）
	//映射函数是一个累积分布函数
	equalizeHist(pic_gray, pic_gray);

	timeCount = (double)cvGetTickCount();//用来计算检测人脸算法执行的时间

	//多尺寸检测人脸
	faceCascade.detectMultiScale(
		pic_gray, //Matrix of the type CV_8U containing an image where objects are detected
		faces,//Vector of rectangles where each rectangle contains the detected object（检测到的人脸目标序列）
		1.1,//Parameter specifying how much the image size is reduced at each image scale.
		3,//Parameter specifying how many neighbors each candiate rectangle should have to retain it.
		0
		|CV_HAAR_SCALE_IMAGE,
		//|CV_HAAR_FIND_BIGGEST_OBJECT, 
		//|CV_HAAR_DO_ROUGH_SEARCH,
		//| CV_HAAR_DO_CANNY_PRUNING,
		Size(70, 70)//Minimum possible object size. Objects smaller than that are ignored.
		//Size(300, 300)
		);
	
	timeCount = (double)cvGetTickCount() - timeCount;
	printf("\n detection time = %g ms\n", timeCount / ((double)cvGetTickFrequency()*1000));//打印时间

	for (int i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		ellipse(
			pic,
			center,//Center of the ellipse
			Size(faces[i].width*0.4, faces[i].height*0.6),//Length of the ellipse axes. 
			0, //Ellipse rotation angle in degrees
			0, // Starting angle of the elliptic arc in degrees
			360, //Ending angle of the elliptic arc in degrees
			Scalar(rand() & 255, rand() & 255, rand() & 255), //Ellipse color
			4, //Thickness of the ellipse arc outline, if positive. Otherwise, this indicates that a filled ellipse sector is to be drawn.
			4, //Type of the ellipse boundary.See the line() description.
			0);

		//检测到的人脸是我们的感兴趣(ROI)区域
		//builds matrix from std::vector with or without copying the data
		//template<typename _Tp> explicit Mat(const vector<_Tp>& vec, bool copyData = false);
		Mat faceROI = pic_gray(faces[i]);
		vector<Rect> eyes;//对象eyes中的Rect::x,Rect::y是相对faceROI的位置
		vector<Rect> mouth;
		vector<Rect> nose;

		//在每张检测到的人脸上检测双眼
		eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(2, 2));
		//mouthCascade.detectMultiScale(faceROI, mouth, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(3, 3));
		noseCascade.detectMultiScale(faceROI, nose, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(2, 2));
		//对眼睛标记
		for (int j = 0; j < eyes.size(); j++)
		{
			Point center(faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5);
			//ellipse(pic, center, Size(eyes[j].width*0.5, eyes[j].height*0.5), 0, 0, 360, Scalar(rand() & 255, rand() & 255, rand() & 255), 2, 4, 0);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.2);//function:Rounds floating-point number to the nearest integer
			circle(pic,center, radius, Scalar(255, 255, 0), 1, 8, 0);
		}
		//对嘴巴标记
		//for (int k = 0; k < mouth.size(); k++)
		//{
		//	Point center(faces[i].x + mouth[k].x + mouth[k].width*0.5, faces[i].y + mouth[k].y + mouth[k].height*0.5);
		//	ellipse(pic, center, Size(mouth[k].width*0.5, mouth[k].height*0.5), 0, 0, 360, Scalar(255, 255, 0), 2, 4, 0);
		//}
		//对鼻子标记
		for (int z = 0; z < nose.size(); z++)
		{
			Point center(faces[i].x + nose[z].x + nose[z].width*0.5, faces[i].y + nose[z].y + nose[z].height*0.3);
			ellipse(pic, center, Size(nose[z].height*0.3, nose[z].width*0.5), 0, 0, 360, Scalar(0, 255, 0), 1, 8, 0);
		}
	}
	namedWindow(windowName,CV_WINDOW_NORMAL);
	imshow(windowName, pic);
}




int main(int argc, int** argv)
{
	Mat videoFrame;
	VideoCapture videoCapture;
	double detectTime=0.0;

	if ((!faceCascade.load(faceCascadeName))
		|| (!eyeCascade.load(eyeCascadeName))
		|| (!mouthCascade.load(mouthCascadeName))
		|| (!noseCascade.load(noseCascadeName))
		)
	{
		printf("there is something wrong in loading cascade file: *.xml");
		return -2;
	}
	//=======================用于输入图片进行人脸检测=============================
	//printf("\n\t\t\t【FACE DETECT BY ischan】\n");
	//printf("\n\t  if want to cancel this test,press 【ESC】 on the keyboard\n");
	//printf("\n\t\t\t All rights reserved by ischan");

	//Mat img = imread("106.jpg");
	//if (!img.data)
	//{
	//	printf("there is somethingwrong in loading picture!!");
	//	return -1;
	//}

	//detectAndShow(img);
	//waitKey();
	
	//========================================================================


	//=======================用于读取视频流进行人脸检测===========================
	printf("\n\t\t\t【FACE DETECT BY ischan】\n");
	printf("\n\t  if want to cancel this test,press 【ESC】 on the keyboard\n");
	printf("\n\t\t\t All rights reserved by ischan");
	
	videoCapture.open(0);
	if (!videoCapture.isOpened())
	{
		printf("can't open video device!");
	}

	while (videoCapture.read(videoFrame))
	{
		if (videoFrame.empty())
		{
			printf("can't capture frame correctly");
			break;
		}
		detectAndShow(videoFrame);

		int key = waitKey(5);
		if ((char)key == 27) { break; }//ESC的ASCII码为27
	}

	//========================================================================

	return 0;	
}
