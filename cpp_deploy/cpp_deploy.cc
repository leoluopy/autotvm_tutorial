
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <opencv2/opencv.hpp>
#include <sys/time.h>

typedef struct FaceInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    float landmarks[10];
};

static void nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, float nmsthreshold) {
    std::sort(input.begin(), input.end(),
              [](const FaceInfo &a, const FaceInfo &b) {
                  return a.score > b.score;
              });

    int box_num = input.size();
    std::vector<int> merged(box_num, 0);

    for (int i = 0; i < box_num; i++) {
        if (merged[i])
            continue;

        output.push_back(input[i]);

        float h0 = input[i].y2 - input[i].y1 + 1;
        float w0 = input[i].x2 - input[i].x1 + 1;

        float area0 = h0 * w0;

        for (int j = i + 1; j < box_num; j++) {
            if (merged[j])
                continue;

            float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1;//std::max(input[i].x1, input[j].x1);
            float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

            float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2;  //bug fixed ,sorry
            float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

            float inner_h = inner_y1 - inner_y0 + 1;
            float inner_w = inner_x1 - inner_x0 + 1;

            if (inner_h <= 0 || inner_w <= 0)
                continue;
            float inner_area = inner_h * inner_w;
            float h1 = input[j].y2 - input[j].y1 + 1;
            float w1 = input[j].x2 - input[j].x1 + 1;
            float area1 = h1 * w1;
            float score;
            score = inner_area / (area0 + area1 - inner_area);

            if (score > nmsthreshold)
                merged[j] = 1;
        }

    }
}

static std::vector<int> getIds(float *heatmap, int h, int w, float thresh) {
    std::vector<int> ids;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            if (heatmap[i * w + j] > thresh && heatmap[i * w + j] <= 1.0) {
                std::array<int, 2> id = {i, j};
                ids.push_back(i);
                ids.push_back(j);
            }
        }
    }
    return ids;
}

static void decode(cv::Mat &heatmap, cv::Mat &scale, cv::Mat &offset, cv::Mat &landmarks,
                   std::vector<FaceInfo> &faces, int d_w, int d_h, float scale_w, float scale_h,
                   float scoreThresh, float nmsThresh) {
    int fea_h = heatmap.size[2];
    int fea_w = heatmap.size[3];
    int spacial_size = fea_w * fea_h;

    float *heatmap_ = (float *) (heatmap.data);

    float *scale0 = (float *) (scale.data);
    float *scale1 = scale0 + spacial_size;

    float *offset0 = (float *) (offset.data);
    float *offset1 = offset0 + spacial_size;
    float *lm = (float *) landmarks.data;

    std::vector<int> ids = getIds(heatmap_, fea_h, fea_w, scoreThresh);
//    std::cout << ids.size() << std::endl;

    std::vector<FaceInfo> faces_tmp;
    for (int i = 0; i < ids.size() / 2; i++) {
        int id_h = ids[2 * i];
        int id_w = ids[2 * i + 1];
        int index = id_h * fea_w + id_w;

        float s0 = std::exp(scale0[index]) * 4;
        float s1 = std::exp(scale1[index]) * 4;
        float o0 = offset0[index];
        float o1 = offset1[index];

        float x1 = std::max(0., (id_w + o1 + 0.5) * 4 - s1 / 2);
        float y1 = std::max(0., (id_h + o0 + 0.5) * 4 - s0 / 2);
        float x2 = 0, y2 = 0;
        x1 = std::min(x1, (float) d_w);
        y1 = std::min(y1, (float) d_h);
        x2 = std::min(x1 + s1, (float) d_w);
        y2 = std::min(y1 + s0, (float) d_h);
        FaceInfo facebox;
        facebox.x1 = x1;
        facebox.y1 = y1;
        facebox.x2 = x2;
        facebox.y2 = y2;
        facebox.score = heatmap_[index];
        float box_w = x2 - x1;
        float box_h = y2 - y1;
        for (int j = 0; j < 5; j++) {
            facebox.landmarks[2 * j] = x1 + lm[(2 * j + 1) * spacial_size + index] * s1;
            facebox.landmarks[2 * j + 1] = y1 + lm[(2 * j) * spacial_size + index] * s0;
        }
        faces_tmp.push_back(facebox);
    }
    nms(faces_tmp, faces, nmsThresh);
    for (int k = 0; k < faces.size(); k++) {
        faces[k].x1 *= scale_w;
        faces[k].y1 *= scale_h;
        faces[k].x2 *= scale_w;
        faces[k].y2 *= scale_h;

        for (int kk = 0; kk < 5; kk++) {
            faces[k].landmarks[2 * kk] *= scale_w;
            faces[k].landmarks[2 * kk + 1] *= scale_h;
        }
    }
}

class TVMCenterFace {
public:
    TVMCenterFace(const std::string lib_path = "../../centerFace_relay.so") {
        DLDevice dev{kDLGPU, 0};

        // for windows , the suffix should be dll
        mod_factory = tvm::runtime::Module::LoadFromFile(lib_path, "so");
        gmod = mod_factory.GetFunction("default")(dev);
        set_input = gmod.GetFunction("set_input");
        get_output = gmod.GetFunction("get_output");
        run = gmod.GetFunction("run");

        // Use the C++ API
        x = tvm::runtime::NDArray::Empty({1, 3, 544, 960}, DLDataType{kDLFloat, 32, 1}, dev);
        heatmap_gpu = tvm::runtime::NDArray::Empty({1, 1, 136, 240}, DLDataType{kDLFloat, 32, 1}, dev);
        scale_gpu = tvm::runtime::NDArray::Empty({1, 2, 136, 240}, DLDataType{kDLFloat, 32, 1}, dev);
        offset_gpu = tvm::runtime::NDArray::Empty({1, 2, 136, 240}, DLDataType{kDLFloat, 32, 1}, dev);
        lms_gpu = tvm::runtime::NDArray::Empty({1, 10, 136, 240}, DLDataType{kDLFloat, 32, 1}, dev);
    }

    int inference(cv::Mat frame, std::vector<FaceInfo> &faces, bool visulise = false) {
        try {
            faces.clear();
            int h = frame.rows;
            int w = frame.cols;
            float img_h_new = int(ceil(h / 32) * 32);
            float img_w_new = int(ceil(w / 32) * 32);
            float scale_h = img_h_new / float(h);
            float scale_w = img_w_new / float(w);
            cv::Mat input_tensor = cv::dnn::blobFromImage(frame, 1.0, cv::Size(img_w_new, img_h_new),
                                                          cv::Scalar(0, 0, 0),
                                                          true,
                                                          false, CV_32F);
            x.CopyFromBytes(input_tensor.data, 1 * 3 * 544 * 960 * sizeof(float));

            set_input("input0", x);
            timeval t0, t1;
            gettimeofday(&t0, NULL);
            run();
            gettimeofday(&t1, NULL);
            printf("inference cost: %f \n", t1.tv_sec - t0.tv_sec + (t1.tv_usec - t0.tv_usec) / 1000000.);
            get_output(0, heatmap_gpu);
            get_output(1, scale_gpu);
            get_output(2, offset_gpu);
            get_output(3, lms_gpu);
            tvm::runtime::NDArray heatmap_cpu = heatmap_gpu.CopyTo(DLDevice{kDLCPU, 0});
            tvm::runtime::NDArray scale_cpu = scale_gpu.CopyTo(DLDevice{kDLCPU, 0});
            tvm::runtime::NDArray offset_cpu = offset_gpu.CopyTo(DLDevice{kDLCPU, 0});
            tvm::runtime::NDArray lms_cpu = lms_gpu.CopyTo(DLDevice{kDLCPU, 0});


            int dimensionHeatMap[] = {1, 1, h / 4, w / 4};
            cv::Mat heatmap(4, dimensionHeatMap, CV_32F);
            heatmap.data = (uchar *) heatmap_cpu->data;

            int dimensionScale[] = {1, 2, h / 4, w / 4};
            cv::Mat scale(4, dimensionScale, CV_32F);
            scale.data = (uchar *) scale_cpu->data;

            int dimensionOffset[] = {1, 2, h / 4, w / 4};
            cv::Mat offset(4, dimensionOffset, CV_32F);
            offset.data = (uchar *) offset_cpu->data;

            int dimensionLandMark[] = {1, 10, h / 4, w / 4};
            cv::Mat landmark(4, dimensionLandMark, CV_32F);
            landmark.data = (uchar *) lms_cpu->data;

            decode(heatmap, scale, offset, landmark, faces, w, h, scale_w, scale_h, 0.5, 0.3);
            if (visulise) {
                for (int i = 0; i < faces.size(); i++) {
                    cv::rectangle(frame, cv::Point(faces[i].x1, faces[i].y1), cv::Point(faces[i].x2, faces[i].y2),
                                  cv::Scalar(0, 255, 0), 1);
                    for (int j = 0; j < 5; j++) {
                        cv::circle(frame, cv::Point(faces[i].landmarks[2 * j], faces[i].landmarks[2 * j + 1]), 1,
                                   cv::Scalar(255, 255, 0), 1);
                    }
                }
                cv::imshow("out", frame);
                cv::waitKey(0);
            }
            return 0;
        }
        catch (std::exception ex) {
            std::cout << " TVM inference exception: " << ex.what() << std::endl;
            return 1;
        }
    }

private:
    // models
    tvm::runtime::Module mod_factory;
    tvm::runtime::Module gmod;
    tvm::runtime::PackedFunc set_input;
    tvm::runtime::PackedFunc get_output;
    tvm::runtime::PackedFunc run;

    // datas
    tvm::runtime::NDArray x;
    tvm::runtime::NDArray heatmap_gpu;
    tvm::runtime::NDArray scale_gpu;
    tvm::runtime::NDArray offset_gpu;
    tvm::runtime::NDArray lms_gpu;
    tvm::runtime::NDArray yaw_gpu;
    tvm::runtime::NDArray pitch_gpu;
    tvm::runtime::NDArray roll_gpu;
};

#include <dirent.h>

static void getAllFilesPath(std::string path, std::vector<std::string> &filesPath) {
    DIR *dir = opendir(path.c_str());
    if (dir == nullptr) {
        closedir(dir);
        return;
    }
    dirent *ptr = NULL;
    while ((ptr = readdir(dir)) != NULL) {
        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)
            continue;
        if (ptr->d_type & DT_DIR) {
            std::string childPath = path + "/" + std::string(ptr->d_name);
            getAllFilesPath(childPath, filesPath);
        } else {
            std::string filePath = path + "/" + std::string(ptr->d_name);
//            cout<<filePath<<endl;
            if (filePath.substr(filePath.size() - 4, 4) == ".jpg" ||
                filePath.substr(filePath.size() - 4, 4) == ".bmp" ||
                filePath.substr(filePath.size() - 4, 4) == ".png") {
                filesPath.push_back(filePath);
            }
        }
    }
    closedir(dir);
}

static cv::Mat padding_image(cv::Mat &image) {
    float w_h_rate = float(image.cols) / float(image.rows);
    float desired_rate = 960.0 / 544.0;
    if (w_h_rate > desired_rate) {
        // padding vertical
        int padding_size = int((float(image.cols) / desired_rate) - image.rows);
        cv::vconcat(image, cv::Mat::zeros(padding_size, image.cols, CV_8UC3), image);
    } else {
        // padding horizontal
        int padding_size = int(float(image.rows) * desired_rate - image.cols);
        cv::hconcat(image, cv::Mat::zeros(image.rows, padding_size, CV_8UC3), image);
    }
    cv::resize(image, image, cv::Size(960, 544));
    return image;
}

int main(int argc, char **argv) {

    TVMCenterFace centerface = TVMCenterFace();

    std::vector<FaceInfo> faces;
    std::vector<std::string> filesPath;
    getAllFilesPath("../../ims/", filesPath);

    for (int i = 0; i < filesPath.size(); i++) {
        cv::Mat frame = cv::imread(filesPath[i]);
        padding_image(frame);
        centerface.inference(frame, faces, true);
    }

    return 0;
}
