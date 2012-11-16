#include <cv.h>
#include <highgui.h>

using namespace cv;

int main(int argc, char** argv){
    Mat rgb, gray;
    rgb = imread("/usr/local/share/OpenCV/samples/c/fruits.jpg", 1);
    cvtColor(rgb, gray, CV_BGR2GRAY);

    namedWindow("Disp img", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
    imshow("Disp img", gray);

    waitKey(0);

    return 0;
}
