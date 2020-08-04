#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <time.h>
#include <algorithm>

#define SCALE 1.0

using namespace cv;

int vibe(int range, double ratio_min, double ratio_max)
{
    return (ratio_min * range) + (rand() % (int)((ratio_max - ratio_min) * range));
}

Rect *getRoi(Mat *img, CvRect *faceRect)
{

    int x = (faceRect->x + vibe(faceRect->width, -0.2, 0.1)) * SCALE;
    int y = (faceRect->y + vibe(faceRect->height, -0.2, 0.2)) * SCALE;
    int w = vibe(faceRect->width, 0.8, 1.3) * SCALE;
    int h = vibe(faceRect->height, 0.7, 1.1) * SCALE;

    x = min(max(x, 0), img->cols);
    y = min(max(y, 0), img->rows);
    w = min(max(w, 0), img->cols - x - 1);
    h = min(max(h, 0), img->rows - y - 1);

    return new Rect(x, y, w, h);
}

Mat glitch(Mat img, std::vector<uchar> *buf, std::vector<uchar> *buf2, std::vector<int> *params)
{

    std::vector<uchar>::iterator it, it2;

    imencode(".jpg", img, *buf, *params); // encode to jpg

    buf2->resize(buf->size());
    for (it = buf->begin(), it2 = buf2->begin(); it != buf->end(); ++it, ++it2)
    {
        if (*it == 'x' || *it == 't')
        {
            *it2 = rand() % 10;
        }
        else
        {
            *it2 = *it;
        }
    }

    /// decode
    return imdecode(Mat(*buf2), 1);
}

inline CvSeq *getFaces(IplImage *frame_copy, CvHaarClassifierCascade *cvHCC, CvMemStorage *cvMStr)
{

    // グレースケール化, ヒストグラムの均一化
    IplImage *gray = cvCreateImage(cvSize(frame_copy->width, frame_copy->height), IPL_DEPTH_8U, 1);
    IplImage *detect_frame = cvCreateImage(cvSize((frame_copy->width / SCALE),
                                                  (frame_copy->height / SCALE)),
                                           IPL_DEPTH_8U, 1);
    cvCvtColor(frame_copy, gray, CV_BGR2GRAY);
    cvResize(gray, detect_frame, CV_INTER_LINEAR);
    cvEqualizeHist(detect_frame, detect_frame);

    return cvHaarDetectObjects(detect_frame, cvHCC, cvMStr, 1.1, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(30, 30));
}

int main(int argc, char **argv)
{
    // Get input file name
    if (argc != 2)
    {
        std::cout << "Usage: " << std::endl
                  << "  $ glitch [INPUT_FILE]" << std::endl;
        return -1;
    }
    char *input_file = argv[1];

    CvCapture *capture = NULL; // capture object
    IplImage *frame = 0;       // キャプチャ画像用ポインタ
    IplImage *frame_copy = 0;  // キャプチャ画像のコピー用ポインタ
    int input_key;

    // buffer for encode/decode
    std::vector<uchar> buf;
    std::vector<uchar> buf2;
    std::vector<int> params(2);
    params[0] = CV_IMWRITE_JPEG_QUALITY;
    params[1] = 10;

    srand((unsigned int)time(NULL));

    CvHaarClassifierCascade *cvHCC = (CvHaarClassifierCascade *)cvLoad("haarcascade_frontalface_default.xml");
    CvMemStorage *cvMStr = cvCreateMemStorage(0);
    CvSeq *face;

    capture = cvCaptureFromFile(input_file); // ファイル読み込み

    namedWindow("Video", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO); // ウィンドウ作成
    cvMoveWindow("Video", 300, 200);

    // main loop
    while (1)
    {
        frame = cvQueryFrame(capture);
        if (frame == NULL)
            break;

        // フレームコピー用イメージ生成
        frame_copy = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, frame->nChannels);
        if (frame->origin == IPL_ORIGIN_TL)
        {
            cvCopy(frame, frame_copy);
        }
        else
        {
            cvFlip(frame, frame_copy);
        }

        // 関数にすると動かない……
        // face = getFaces(frame_copy, cvHCC, cvMStr);            // face detect

        // 読み込んだ画像のグレースケール化、及びヒストグラムの均一化を行う
        IplImage *gray = cvCreateImage(cvSize(frame_copy->width, frame_copy->height), IPL_DEPTH_8U, 1);
        IplImage *detect_frame = cvCreateImage(cvSize((frame_copy->width / SCALE), (frame_copy->height / SCALE)), IPL_DEPTH_8U, 1);
        cvCvtColor(frame_copy, gray, CV_BGR2GRAY);
        cvResize(gray, detect_frame, CV_INTER_LINEAR);
        cvEqualizeHist(detect_frame, detect_frame);

        // 画像中から検出対象の情報を取得する
        face = cvHaarDetectObjects(detect_frame, cvHCC, cvMStr, 1.1, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(30, 30));

        Mat img(frame_copy);                                  // convert frame to Mat
        Mat img_glitched = glitch(img, &buf, &buf2, &params); // glitch

        // 検出した位置にglitch 画像を貼り付ける
        for (int i = 0; i < face->total; i++)
        {
            CvRect *faceRect = (CvRect *)cvGetSeqElem(face, i);
            /*
            // roiの範囲で、img_glitched を img に貼り付け
            Rect* roi_src = getRoi(&img, faceRect);
            Rect* roi_dst = getRoi(&img, faceRect);
            Mat mat_src = img_glitched(*roi_src);
            Mat mat_dst = img(*roi_dst);

            resize(mat_src, mat_dst, mat_dst.size(), INTER_CUBIC);
*/
            // roiの範囲で、img_glitched を img に貼り付け
            Rect *roi = getRoi(&img, faceRect);
            Mat mat_src = img_glitched(*roi);
            Mat mat_dst = img(*roi);

            mat_src.copyTo(mat_dst);

            delete (roi);
        }

        imshow("Video", img); // show img

        input_key = cvWaitKey(10); // 終了キー入力待ち（タイムアウト付き）
        if (input_key == 'e')
        {
            break;
        }
    }

    cvReleaseCapture(&capture); // delete capture
    cvDestroyWindow("Capture"); // delete window

    cvReleaseMemStorage(&cvMStr);
    cvReleaseHaarClassifierCascade(&cvHCC);

    return 0;
}
