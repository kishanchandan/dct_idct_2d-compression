#include <iostream>
#include <opencv2/opencv.hpp>
#define _USE_MATH_DEFINES
#include <math.h>

using namespace cv;
using namespace std;

Mat HSI(Mat image) {			// Function to convert the image to HSI domain
	float red, green, blue, hue, saturation, intensity;
	Mat hsi_image(image.rows, image.cols, image.type());
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			blue = image.at<Vec3b>(i, j)[0];
			green = image.at<Vec3b>(i, j)[1];
			red = image.at<Vec3b>(i, j)[2];

			intensity = (blue + green + red) / 3;

			int min_val = 0;
			min_val = std::min(red, std::min(blue, green));

			saturation = 1 - 3 * (min_val / (blue + green + red));
			if (saturation < 0.00001)
			{
				saturation = 0;
			}
			else if (saturation > 0.99999) {
				saturation = 1;
			}

			if (saturation != 0)
			{
				hue = 0.5 * ((red - green) + (red - blue)) / sqrt(((red - green)*(red - green)) + ((red - blue)*(green - blue)));
				hue = acos(hue);

				if (blue <= green)
				{
					hue = hue;
				}
				else {
					hue = ((360 * 3.14159265) / 180.0) - hue;
				}
			}

			//hsi.at<Vec3b>(i, j)[0] = (h * 180) / 3.14159265;
			//hsi.at<Vec3b>(i, j)[1] = s * 100;
			//hsi.at<Vec3b>(i, j)[0] = in;
			//hsi.at<Vec3b>(i, j)[1] = in;
			hsi_image.at<Vec3b>(i, j)[2] = intensity;

		}
	}
	return hsi_image;
}

Mat CALCULATE_DCT(Mat hsi) {
	Mat dct_image(hsi.rows, hsi.cols, hsi.type());
	int height = hsi.rows - (hsi.rows % 8);		// To get the exact bound
	int width = hsi.cols - (hsi.cols % 8);
	float temp;

	for (int m = 0; m < height; m += 8)			// Further dividing into blocks
	{
		for (int n = 0; n < width; n += 8)
		{
			for (int i = m; i < m + 8; i++)
			{
				for (int j = n; j < n + 8; j++)
				{
					double Ci, Cj;
					temp = 0.0;
					if (i == 0) Ci = 1.0 / sqrt(2.0); else Ci = 1.0;
					if (j == 0) Cj = 1.0 / sqrt(2.0); else Cj = 1.0;
					for (int x = m; x < m + 8; x++)
					{
						for (int y = n; y < n + 8; y++)
						{
							temp += (hsi.at<Vec3b>(x, y)[2]) *(cos((((2 * x) + 1)) * ((i * 3.14159)) / (2 * 8))) *(cos((((2 * y) + 1)) * ((j * 3.14159)) / (2 * 8)));
						}
					}
					temp *= ((2 / sqrt(8 * 8)))*(Ci * Cj);
					dct_image.at<Vec3b>(i, j)[2] = int(temp);				// To finally add the value to dct_image matrix
				}
			}
		}
	}

	return dct_image;

}

Mat CALCULATE_DCT_DC_PART(Mat dct) {

	int height = dct.rows - (dct.rows % 8);
	int width = dct.cols - (dct.cols % 8);
	float temp;
	Mat dct_dc_image(dct.rows, dct.cols, dct.type());
	dct_dc_image = dct.clone();
	for (int m = 0; m < height; m += 8)
	{
		for (int n = 0; n < width; n += 8)
		{
			for (int i = m; i < m + 8; i++)
			{
				for (int j = n; j < n + 8; j++)
				{
					double Ci, Cj;
					temp = 0.0;
					if (i == 0) Ci = 1.0 / sqrt(2.0); else Ci = 1;
					if (j == 0) Cj = 1.0 / sqrt(2.0); else Cj = 1;
					for (int x = m; x < m + 8; x++)
					{
						for (int y = n; y < n + 8; y++)
						{
							if (i == m && j == n)
							{
								dct_dc_image.at<Vec3b>(i, j)[2] = dct_dc_image.at<Vec3b>(i, j)[2];
							}
							else
							{
								dct_dc_image.at<Vec3b>(i, j)[2] = 0;
							}
						}
					}

				}
			}
		}
	}
	return dct_dc_image;
}
Mat CALCULATE_DCT_9_DC_PARTS(Mat dct) {
	Mat freqimg(dct.rows, dct.cols, dct.type());
	Mat dcts(dct.rows, dct.cols, dct.type());
	Mat d1(dct.rows, dct.cols, dct.type());

	dcts = dct.clone();

	int height = dct.rows - (dct.rows % 8);
	int width = dct.cols - (dct.cols % 8);
	float temp;

	Mat dct_9dc_image(dct.rows, dct.cols, dct.type());
	dct_9dc_image = dct.clone();
	for (int m = 0; m < height; m += 8)
	{
		for (int n = 0; n < width; n += 8)
		{
			for (int i = m; i < m + 8; i++)
			{
				for (int j = n; j < n + 8; j++)
				{
					double Ci, Cj;
					temp = 0.0;
					if (i == 0) Ci = 1.0 / sqrt(2.0); else Ci = 1;
					if (j == 0) Cj = 1.0 / sqrt(2.0); else Cj = 1;
					for (int x = m; x < m + 8; x++)
					{
						for (int y = n; y < n + 8; y++)
						{
							if (i == m && j == n || i == m && j == n + 1 || i == m && j == n + 2 || i == m && j == n + 3 || i == m + 1 && j == n || i == m + 1 && j == n + 1 || i == m + 1 && j == n + 2 || i == m + 2 && j == n || i == m + 2 && j == n + 1)
							{
								dct_9dc_image.at<Vec3b>(i, j)[2] = dct_9dc_image.at<Vec3b>(i, j)[2];
							}
							else
							{
								dct_9dc_image.at<Vec3b>(i, j)[2] = 0;
							}
						}
					}

				}
			}
		}
	}
	return dct_9dc_image;
}

Mat CALCULATE_IDCT(Mat dct) {
	int height = dct.rows - (dct.rows % 8);
	int width = dct.cols - (dct.cols % 8);
	float temp;
	Mat idct_dc_image(dct.rows, dct.cols, dct.type());
	for (int m = 0; m < height; m += 8)
	{
		for (int n = 0; n < width; n += 8)
		{
			for (int i = m; i < m + 8; i++)
			{
				for (int j = n; j < n + 8; j++)
				{
					double Ci, Cj;
					temp = 0.0;
					if (i == 0) Ci = 1.0 / sqrt(2.0); else Ci = 1.0;
					if (j == 0) Cj = 1.0 / sqrt(2.0); else Cj = 1.0;
					for (int x = m; x < m + 8; x++)
					{
						for (int y = n; y < n + 8; y++)
						{

							temp += (dct.at<Vec3b>(x, y)[2]) *(cos((((2 * i) + 1)) * ((x * 3.14159)) / (2 * 8))) *(cos((((2 * j) + 1)) * ((y * 3.14159)) / (2 * 8))) *(Ci*Cj);
						}
					}
					temp *= ((2 / sqrt(8 * 8)));
					idct_dc_image.at<Vec3b>(i, j)[2] = int(temp);				//Finally adding the pixel values to idct_image
				}
			}
		}
	}


	return idct_dc_image;
}


int main()
{
	std::ofstream ofs1, ofs2;
	ofs1.open("test.txt", std::ofstream::out | std::ofstream::app);
	ofs2.open("testfreqimg.txt", std::ofstream::out | std::ofstream::app);


	Mat image;

	// LOAD image
	image = imread("basel3.bmp", CV_LOAD_IMAGE_COLOR);
	if (image.empty())
		cerr << "Error: Loading image" << endl;

	Mat hsi(image.rows, image.cols, image.type());
	Mat dct_image(hsi.rows, hsi.cols, hsi.type());
	Mat dct_dc_image(hsi.rows, hsi.cols, hsi.type());
	Mat dct_9dc_image(hsi.rows, hsi.cols, hsi.type());
	Mat idct_dc_image(hsi.rows, hsi.cols, hsi.type());
	Mat idct_9dc_image(hsi.rows, hsi.cols, hsi.type());
	hsi = HSI(image);  //Conversion of image to HSI

	dct_image = CALCULATE_DCT(hsi);  // Creates a DCT image

	dct_dc_image = CALCULATE_DCT_DC_PART(dct_image); // Keeps only DC components in an image

	dct_9dc_image = CALCULATE_DCT_9_DC_PARTS(dct_image);  // Keeps the first 9 lowest frequencies

	idct_dc_image = CALCULATE_IDCT(dct_dc_image);  // IDCT OF DC COMPONENTS

	idct_9dc_image = CALCULATE_IDCT(dct_9dc_image); // IDCGT OF first 9 LOWEST FREQUENCY COMPONENTS


													//namedWindow("AutoDCT image", CV_WINDOW_AUTOSIZE);



	namedWindow("IDCT(9DC) image", CV_WINDOW_AUTOSIZE);
	namedWindow("IDCT(DC) image", CV_WINDOW_AUTOSIZE);
	namedWindow("Only9DC image", CV_WINDOW_AUTOSIZE);
	namedWindow("OnlyDC image", CV_WINDOW_AUTOSIZE);
	namedWindow("DCT image", CV_WINDOW_AUTOSIZE);
	namedWindow("HSI image", CV_WINDOW_AUTOSIZE);
	namedWindow("RGB image", CV_WINDOW_AUTOSIZE);
	//vector<Mat> planes;
	//split(hsi, planes);
	/*vector<Mat> outplanes(planes.size());
	//for (size_t i = 0; i < planes.size(); i++)
	//{
	//planes[i].convertTo(planes[i], CV_32FC1);
	//dct(planes[i], outplanes[i]);
	//}*/
	//vector<Mat> mergedplanes(planes.size());
	//Mat merged, merged1, merged2, merged3,mergedidct;
	//planes[0].convertTo(mergedplanes[0], CV_32F);
	//planes[1].convertTo(mergedplanes[1], CV_32F);
	//planes[2].convertTo(mergedplanes[2], CV_32F);
	//dct(mergedplanes[0], mergedplanes[0]);
	//dct(mergedplanes[1], mergedplanes[1]);
	//dct(mergedplanes[2], mergedplanes[2]);
	////merged = mergedplanes[2];
	//merge(mergedplanes, merged);

	////dct(merged,mergedidct, DCT_INVERSE);
	////mergedidct.convertTo(mergedidct, CV_8U);




	//imshow("AutoDCT image", merged);



	imshow("IDCT(9DC) image", idct_9dc_image);
	imshow("IDCT(DC) image", idct_dc_image);
	imshow("Only9DC image", dct_9dc_image);
	imshow("OnlyDC image", dct_dc_image);
	imshow("DCT image", dct_image);
	imshow("HSI image", hsi);
	imshow("RGB image", image);

	waitKey(0);
	return 0;
}


