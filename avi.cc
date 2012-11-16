#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <time.h>
#include <algorithm>

#define SCALE 4.0

using namespace cv;

int vibe(int range, double ratio_min, double ratio_max){
    return (ratio_min * range) + (rand() % (int)((ratio_max-ratio_min)*range));
}

Rect* getRoi(Mat* img, CvRect* faceRect){
/*                
    int x = (faceRect->x + vibe(faceRect->width, -0.3, 0.3)) * SCALE;
    int y = (faceRect->y + vibe(faceRect->height, -0.2, 0.3)) * SCALE;
    int w = vibe(faceRect->width, 0.8, 1.4) * SCALE;
    int h = vibe(faceRect->height, 0.8, 1.2) * SCALE;

    x = min(max(x, 0), img->cols);
    y = min(max(y, 0), img->rows);
    w = min(max(w, -x), img->cols - x);
    h = min(max(h, -y), img->rows - y);
*/


    int x = faceRect->x * SCALE;
    int y = faceRect->y * SCALE;
    int w = faceRect->width * SCALE;
    int h = faceRect->height * SCALE;

    Rect* r = new Rect(x, y, w, h);
    return r;
//    return new Rect(x,y,w,h);
}

Mat glitch(Mat img, std::vector<uchar>* buf, std::vector<uchar>* buf2, std::vector<int>* params){

    std::vector<uchar>::iterator it, it2;

    imencode(".jpg", img, *buf, *params);    // encode to jpg

    buf2->resize( buf->size());
    for(it = buf->begin(), it2 = buf2->begin(); it != buf->end(); ++it, ++it2){
        if(*it == '0' || *it == 'x'){
            *it2 = rand()%10;
        }else{
            *it2 = *it;
        }
    }

    /// decode
    return imdecode(Mat(*buf2), 1);
}

inline CvSeq* getFaces(IplImage* frame_copy, CvHaarClassifierCascade* cvHCC, CvMemStorage* cvMStr){

    // グレースケール化, ヒストグラムの均一化
    IplImage* gray = cvCreateImage(cvSize(frame_copy->width, frame_copy->height), IPL_DEPTH_8U, 1);
    IplImage* detect_frame = cvCreateImage(cvSize((frame_copy->width / SCALE),
                                                  (frame_copy->height / SCALE)),
                                           IPL_DEPTH_8U, 1);
    cvCvtColor(frame_copy, gray, CV_BGR2GRAY);
    cvResize(gray, detect_frame, CV_INTER_LINEAR);
    cvEqualizeHist(detect_frame, detect_frame);
    
    return cvHaarDetectObjects(detect_frame, cvHCC, cvMStr, 1.1, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(30, 30));
}

int main(int argc, char **argv){

    CvCapture *capture = NULL;  // capture object
    IplImage *frame = 0;        // キャプチャ画像用ポインタ
    IplImage *frame_copy = 0;   // キャプチャ画像のコピー用ポインタ
    int input_key;

    // buffer for encode/decode
    std::vector<uchar> buf;
    std::vector<uchar> buf2;
    std::vector<int> params(2);
    params[0] = CV_IMWRITE_JPEG_QUALITY;
    params[1] = 10;

    srand((unsigned int)time(NULL));
    
    CvHaarClassifierCascade* cvHCC = (CvHaarClassifierCascade*)cvLoad("haarcascade_frontalface_default.xml");
    CvMemStorage* cvMStr = cvCreateMemStorage(0);
    CvSeq* face;
    
    capture = cvCaptureFromFile("/home/fand/cv/tes.avi");              // ファイル読み込み

    namedWindow("Video", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);      // ウィンドウ作成
    cvMoveWindow("Video", 300, 200);

    // main loop
    while (1) {
        frame = cvQueryFrame (capture);
        if(frame == NULL) break;


        // フレームコピー用イメージ生成
        frame_copy = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, frame->nChannels);
        if(frame->origin == IPL_ORIGIN_TL) {
            cvCopy(frame, frame_copy);
        } else {
            cvFlip(frame, frame_copy);
        }


        
//        face = getFaces(frame_copy, cvHCC, cvMStr);            // face detect

        // 読み込んだ画像のグレースケール化、及びヒストグラムの均一化を行う
        IplImage* gray = cvCreateImage(cvSize(frame_copy->width, frame_copy->height), IPL_DEPTH_8U, 1);
        IplImage* detect_frame = cvCreateImage(cvSize((frame_copy->width / SCALE), (frame_copy->height / SCALE)), IPL_DEPTH_8U, 1);
        cvCvtColor(frame_copy, gray, CV_BGR2GRAY);
        cvResize(gray, detect_frame, CV_INTER_LINEAR);
        cvEqualizeHist(detect_frame, detect_frame);

        // 画像中から検出対象の情報を取得する
        face = cvHaarDetectObjects(detect_frame, cvHCC, cvMStr, 1.1, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(30, 30) );

        

        Mat img(frame_copy);                                   // convert frame to Mat
        
        Mat img_glitched = glitch(img, &buf, &buf2, &params);  // glitch
/*
        std::vector<uchar>::iterator it, it2;

        imencode(".jpg", img, buf, params);    // encode to jpg

//        buf2.resize( buf.size());
        for(it = buf.begin(), it2 = buf2.begin(); it != buf.end(); ++it, ++it2){
            if(*it == '0' || *it == 'x'){
                *it2 = rand()%10;
            }else{
                *it2 = *it;
            }
        }
        
        /// decode
        return imdecode(Mat(*buf2), 1);
*/
        








        
        // 検出した位置にglitch 画像を貼り付ける
        for (int i = 0; i < face->total; i++) {
            CvRect* faceRect = (CvRect*)cvGetSeqElem(face, i);

            // roiの範囲で、img_glitched を img に貼り付け
            Rect* roi = getRoi(&img, faceRect);



            
            Mat src_roi = img_glitched(*roi);
            Mat dst_roi = img(*roi);
            src_roi.copyTo(dst_roi);

            delete(roi);
        }
        
        imshow("Video", img);    // show img

        input_key = cvWaitKey (10); // 終了キー入力待ち（タイムアウト付き）
        if (input_key == 'e') {
            break;
        }
    }
    
    cvReleaseCapture(&capture);  // delete capture
    cvDestroyWindow("Capture");  // delete window

    cvReleaseMemStorage(&cvMStr);
    cvReleaseHaarClassifierCascade(&cvHCC);

    return 0;
}
