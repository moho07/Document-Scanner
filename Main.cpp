#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat frame;

float width_A4 = 420, height_A4 = 596;

Mat preprocessing(Mat img)
{
	Mat imggray, imgblur, imgcanny, imgdil;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2));

	cvtColor(img, imggray, COLOR_BGR2GRAY);
	GaussianBlur(imggray, imgblur, Size(3, 3), 5, 0);
	Canny(imgblur, imgcanny, 50, 150);
	dilate(imgcanny, imgdil, kernel);

	return imgdil;
}

vector<Point> getCorners(Mat framedil)
{
	vector<Point> points;

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	
	findContours(framedil, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	vector<vector<Point>> cp(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		int area = contourArea(contours[i]);
		if (area > 2500)
		{
			float peri = arcLength(contours[i], true);
			approxPolyDP(contours[i], cp[i], 0.02 * peri, true);

			if (cp[i].size() == 4)
				points = cp[i];
		}
	}

	return points;

}

vector<Point> reorder(vector<Point> points)
{
	vector<int> sumpoints;
	vector<int> diffpoints;
	vector<Point> finalpoints;

	for (int i = 0; i < points.size(); i++)
	{
		sumpoints.push_back(points[i].x + points[i].y);
		diffpoints.push_back(points[i].x - points[i].y);
	}

	finalpoints.push_back(points[min_element(sumpoints.begin(), sumpoints.end()) - sumpoints.begin()]); //0
	finalpoints.push_back(points[max_element(diffpoints.begin(), diffpoints.end()) - diffpoints.begin()]); //1
	finalpoints.push_back(points[min_element(diffpoints.begin(), diffpoints.end()) - diffpoints.begin()]); //2
	finalpoints.push_back(points[max_element(sumpoints.begin(), sumpoints.end()) - sumpoints.begin()]); //3

	return finalpoints;
}

Mat warp(vector<Point> corners)
{
	Mat framewarp;

	Point2f src[4];

	for (int i = 0; i < 4; i++)
		src[i] = corners[i];

	Point2f dst[4] = { {0.0f, 0.0f}, {width_A4, 0.0f}, {0.0f, height_A4}, {width_A4, height_A4} };

	Mat matrix = getPerspectiveTransform(src, dst);
	warpPerspective(frame, framewarp, matrix, Size(width_A4, height_A4));

	return framewarp;
}

Mat ScanDocument()
{
	Mat framewarp, finalimg, framewarpgray;

	vector<Point> points;
	points = getCorners(preprocessing(frame));
	points = reorder(points);

	framewarp = warp(points);

	Rect crop(5, 5, width_A4 - 10, height_A4 - 10);

	framewarp = framewarp(crop);

	cvtColor(framewarp, framewarpgray, COLOR_BGR2GRAY);

	adaptiveThreshold(framewarpgray, finalimg, 225, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 23, 3);

	return finalimg;
}

int main()
{
	string path("Folder/img.jpg");

	frame = imread(path);
	
	Mat finalimg;

	finalimg = ScanDocument();

	imshow("Document", finalimg);
	
	waitKey(0);
}
