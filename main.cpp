#include <fstream>
#include <iostream>
#include <iterator>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <vector>
#include <string.h>


using namespace std;
using namespace cv;

const char* dataset_path = ("/home/jan/Documents/autovision_dataset_20171110/20171110_autovisionDataset/");
const char* destination_path = ("/media/jan/My Files/autovision/dataset_export/");

struct yolo_bounding_box{
    int cat;
    float c_x;
    float c_y;
    float w;
    float h;
};

struct bounding_box{
    float a_x;
    float a_y;
    float b_x;
    float b_y;
};

struct point{
    float x;
    float y;
};

//Function to draw bounding box
void bboxDraw(InputOutputArray img,float a_x, float a_y, float b_x, float b_y, Scalar color){
    float l_u_x = a_x;
    float l_u_y = b_y;
    float r_b_x = 2*b_x-a_x;
    float r_b_y = 2*a_y-b_y;
//    Scalar color = ( 0, 255, 255 );

    rectangle(img,Point(l_u_x, l_u_y), Point(r_b_x, r_b_y ), color, 3, 8,0);
//    cout << "\n"<<l_u_x<<" " << l_u_y <<" "<< r_b_x<<" " << r_b_y;
};


Rect transform_bb(int h, bounding_box input, Mat mapx, Mat mapy, InputOutputArray img){
    Mat black_img = Mat(h,h, CV_8UC1,cvScalar(0));
    Mat transformed;
    vector<vector<Point>> cnt;
    vector<Vec4i> hrch;
    bounding_box n_bb; //new bounding box

    bboxDraw(black_img, input.a_x, input.a_y, input.b_x, input.b_y, cvScalar(255,255,255));
    remap(black_img, transformed,mapx, mapy, CV_INTER_LINEAR, BORDER_CONSTANT);
    cv::findContours(transformed, cnt, hrch, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    RNG rng(12345);

    ////////
    Mat drawing = Mat::zeros( black_img.size(), CV_8UC3 );
    for( int i = 0; i< 2; i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( drawing, cnt, i, color, 2, 8, hrch, 0, Point() );
    }
    namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
    imshow( "Contours", drawing );
    ////////

    vector<vector<Point> > cnt_poly( cnt[0].size() );

    approxPolyDP( Mat(cnt[0]), cnt_poly[0], 3, true );
    Rect brct;

    brct = boundingRect(Mat(cnt_poly[0]));

    // for output preview
    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
    rectangle(img, brct.tl(), brct.br(), Scalar(200,0,200), 2, 8, 0 );

//    imshow("img", black_img);
//    waitKey(-1);
    return brct;
}
// Function to get all files from  directory
std::vector<std::string> open(std::string path = dataset_path) {

    DIR*    dir;
    dirent* pdir;
    std::vector<std::string> files;

    dir = opendir(path.c_str());

    while (pdir = readdir(dir)) {
        files.push_back(pdir->d_name);
    }

    return files;
}

/**	Function to get the file extension*/
string getFileExt(const string& s)
{
    size_t i = s.rfind('.', s.length());
    if (i != string::npos)
    {
        return(s.substr(i + 1, s.length() - i));
    }

    return("");
}

std::vector<yolo_bounding_box> get_yolo_bound_boxs(string name){
    name = name.substr(0, 6);
    short loop=0; //short for loop for input

    std::string delimiter = " ";
    std::vector< yolo_bounding_box > arr;

    string line="0"; //this will contain the data read from the file
    ifstream myfile (dataset_path + name + ".txt"); //opening the file.
    if (myfile.is_open()) //if the file is open
    {
        while (! myfile.eof()) //while the end of file is NOT reached
        {
            getline(myfile, line); //get one line from the file
            if(line == "")
                break;
            size_t pos = 0;
            std::string token;
            yolo_bounding_box curr_bb;
            int i = 0;

            // filling bbox struct
            while ((pos = line.find(delimiter)) != std::string::npos) {
                token = line.substr(0, pos);
                switch (i) {
                    case 0:
                        curr_bb.cat = std::stoi(token);
                    case 1:
                        curr_bb.c_x = std::stof(token);
                    case 2:
                        curr_bb.c_y = std::stof(token);
                    case 3:
                        curr_bb.w = std::stof(token);
                }
                line.erase(0, pos + delimiter.length());
                i++;
            }
            curr_bb.h = std::stof(line);
            arr.push_back(curr_bb);
            loop++;
        }
        myfile.close(); //closing the file
        return arr;
    }

    else cout << "Unable to open file"; //if the file is not open output};

}

/**	Function to calculate shift*/
float calc_shift(float x1, float x2, float cx, float k)
{
    float thresh = 1;
    float x3 = x1 + (x2 - x1)*0.5;
    float res1 = x1 + ((x1 - cx)*k*((x1 - cx)*(x1 - cx)));
    float res3 = x3 + ((x3 - cx)*k*((x3 - cx)*(x3 - cx)));

    if (res1>-thresh && res1 < thresh)
        return x1;
    if (res3<0)
    {
        return calc_shift(x3, x2, cx, k);
    }
    else
    {
        return calc_shift(x1, x3, cx, k);
    }
}

float getRadialX(float x, float y, float cx, float cy, float k, bool scale, Vec4f props)
{
    float result;
    if (scale)
    {
        float xshift = props[0];
        float yshift = props[1];
        float xscale = props[2];
        float yscale = props[3];

        x = (x*xscale + xshift);
        y = (y*yscale + yshift);
        result = x + ((x - cx)*k*((x - cx)*(x - cx) + (y - cy)*(y - cy)));
    }
    else
    {
        result = x + ((x - cx)*k*((x - cx)*(x - cx) + (y - cy)*(y - cy)));
    }
    return result;
}

float getRadialY(float x, float y, float cx, float cy, float k, bool scale, Vec4f props)
{
    float result;
    if (scale)
    {
        float xshift = props[0];
        float yshift = props[1];
        float xscale = props[2];
        float yscale = props[3];

        x = (x*xscale + xshift);
        y = (y*yscale + yshift);
        result = y + ((y - cy)*k*((x - cx)*(x - cx) + (y - cy)*(y - cy)));
    }
    else
    {
        result = y + ((y - cy)*k*((x - cx)*(x - cx) + (y - cy)*(y - cy)));
    }
    return result;
}

//deprec
point get_revers_mat(Mat mapx, Mat mapy,float x, float y, int h){

    int i, j,x1,y1;
    float diff, diff1;
    point p;
    diff1 = 1;
    for(i=0; i<h;i++){
        for(j=0; j<h;j++){
            diff = mapx.at<float>(j,i) - x;
            if(abs(diff)<diff1){
                x1=i; //x
//                y1=j;
                diff1 = abs(diff);
            }
        }
    }

    diff1 = 1;
    for(i=0; i<h;i++){
        for(j=0; j<h;j++){
            diff = mapy.at<float>(j,i) - y;
            if(abs(diff)<diff1){
                y1=j; //y
//                x1=i;
                diff1 = abs(diff);
            }
        }
    }
//    cout << "\nx " << x1 << " y "<< y1;
    p.x = x1;
    p.y = y1;
    return p;

}

void rewrYoloBbox(Mat mapx, Mat mapy, string name, int w, int h, Size sz, InputOutputArray img)
{
    ofstream out_bb;
    name = name.substr(0, 6);
    out_bb.open(destination_path+name+".txt");
    int i;
    Mat img1 = img.getMat().clone();
    Rect bbox1;
    bounding_box bbox;
    yolo_bounding_box btyolo;
    std::vector<yolo_bounding_box> yolo_bb;
    yolo_bb = get_yolo_bound_boxs(name);
    i = 0;

    while (i < yolo_bb.size()){
        bbox.a_x = (yolo_bb[i].c_x - yolo_bb[i].w/2) * w;
        bbox.a_y = yolo_bb[i].c_y * h;
        bbox.b_x = yolo_bb[i].c_x * w;
        bbox.b_y = (yolo_bb[i].c_y + yolo_bb[i].h/2) * h;

        bbox1 = transform_bb(h, bbox, mapx, mapy, img1);
        cout << "\nX " << bbox1.x << " Y " << bbox1.y << " width " << bbox1.width << " height " << bbox1.height;

        btyolo.cat = yolo_bb[i].cat;
        btyolo.c_x = float(bbox1.x) / w;
        btyolo.c_y = float(bbox1.y) / h;
        btyolo.w = float(bbox1.width) / w;
        btyolo.h = float(bbox1.height) / h;


        out_bb <<btyolo.cat<<" "<<btyolo.c_x <<" "<<btyolo.c_y<<" "<<btyolo.w<<" "<<btyolo.h<<"\n";

        i++;
    }

    imshow("bounded", img1);
    waitKey(-1);
    out_bb.close();



}
//	Fish Eye Function							Cx, Cy Center of x & y
//	_src : Input image, _dst : Output image, Cx,Cy coordinates from where the distorted image will have as initial point, k : distortion factor
void fishEye(InputArray _src, OutputArray _dst, double Cx, double Cy,
             double k, string name, Size sz, bool scale, InputOutputArray img)
{
    Mat src = _src.getMat();
    Mat mapx = Mat(src.size(), CV_32FC1);
    Mat mapy = Mat(src.size(), CV_32FC1);

    int w = src.cols;	//	Width
    int h = src.rows;	//	Height

    Vec4f props;
    //Calculating x and y shifts to be applied
    float xShift = calc_shift(0, Cx - 1, Cx, k);
    props[0] = xShift;
    float newCenterX = w - Cx;
    float xShift2 = calc_shift(0, newCenterX - 1, newCenterX, k);

    float yShift = calc_shift(0, Cy - 1, Cy, k);
    props[1] = yShift;
    float newCenterY = w - Cy;
    float yShift2 = calc_shift(0, newCenterY - 1, newCenterY, k);

    //	Calculating the scale factor from the x & y shifts accordingly
    float xScale = (w - xShift - xShift2) / w;
    props[2] = xScale;
    float yScale = (h - yShift - yShift2) / h;
    props[3] = yScale;

    float* p = mapx.ptr<float>(0);

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            *p++ = getRadialX((float)x, (float)y, Cx, Cy, k, scale, props);
        }
    }

    p = mapy.ptr<float>(0);

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            *p++ = getRadialY((float)x, (float)y, Cx, Cy, k, scale, props);
        }
    }


    remap(src, _dst, mapx, mapy, CV_INTER_LINEAR, BORDER_CONSTANT);
    rewrYoloBbox(mapx, mapy, name, w, h, sz,img);

}

void proceed(string file_name){
    string ext;
    ext = getFileExt(file_name);	//	Check if the file name provided is of an image
    if (!ext.compare("jpg") || !ext.compare("jpeg") || !ext.compare("png") || !ext.compare("bmp"))
    {
        Mat input_image = imread(dataset_path+file_name, 1);	//	Read the image
        Mat output_image;
        Size sz = input_image.size();

        // Fish Eye Effect looks good in Square shape
        if (sz.height != sz.width)
        {
            cout << "\n Resizing the image to square ";	// Converting to square shape
            double mx = max(sz.height, sz.width);
//            if (mx >= 1000)								// Resizing large images (>1000px) to fit the screen
//            {
//                mx = mx / 2;
//            }
            Size nw_sz;
            nw_sz.height = nw_sz.width = mx;
            cout << nw_sz;
            resize(input_image, input_image, nw_sz, 0, 0, 3);
        }

        fishEye(input_image, output_image, input_image.cols / 2,
                input_image.rows / 2, 0.0001, file_name, sz, true,output_image);	// Fish Eye Function

        imwrite(destination_path+file_name, output_image);

    }

}
/**	Main Function*/
int main()
{
    string ext,file_name, name;
    int i=0;
    std::vector<std::string> f;

    f = open();

    std::fstream file;

    while(i<f.size()){
        file_name = f[i];
        try {
            proceed(file_name);
        }
        catch (...){
            cout<<"not worked on this size of image";
        };
        i++;

    }

     return 0;
}


/*#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <iostream>
#include <string>
using namespace cv;

int main (){
cv::VideoCapture cap("/home/dan/Videos/out/%06d.png");
cv::VideoWriter outputVideo;
outputVideo.open("/home/dan/Videos/AVM.avi", CV_FOURCC('M','J','P','G'), 60, cv::Size(1000,1000),true);

cv::Mat frame;
cap>>frame;
int i =1;
while(!frame.empty()){
   outputVideo<<frame;

   cv::imshow("dfdf",frame);
   cv::waitKey(10);
   std:std::cout<<i++<<std::endl;
   cap>>frame;
}
cap.release();
outputVideo.release();
return 0;
}
*/