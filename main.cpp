#include <cuAprilTags.h>
#include <cuda.h>
#include <vector_types.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <apriltag.h>
#include <tag36h11.h>

#include <vector>
#include <chrono>
#include <filesystem>

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>

typedef std::chrono::steady_clock Clock;
typedef std::chrono::time_point<Clock> Instant;
typedef std::chrono::duration<double, std::milli> Duration;

int main(int argc, char **argv) {
  int err;

  std::cout << "OpenCV version: " << CV_VERSION << std::endl;

  std::vector<std::pair<cv::Mat, cv::Mat>> imgs;

  for (const auto &entry : std::filesystem::directory_iterator("./samples/")) {
  	std::cout << entry.path() << std::endl;
  	
  	cv::Mat img = cv::imread(entry.path());
  	cv::Mat grey;
  	
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::cvtColor(img, grey, cv::COLOR_RGB2GRAY);

    imgs.push_back(std::pair(img, grey));
  }

  int width = imgs[0].first.cols;
  int height = imgs[0].first.rows;

  const int ALLOC_TAGS = 10;
  const int ITERS = 40;
  const int WARMUP = 10;

  // ================= libCuApriltags =================

  cuAprilTagsHandle handle;
  err = nvCreateAprilTagsDetector(&handle, width, height, 4, NVAT_TAG36H11, NULL, 6);
  printf("nvCreateAprilTagsDetector() -> %u\n", err);

  cuAprilTagsImageInput_st input = {0};

  err = cuMemAllocPitch((CUdeviceptr*)(&input.dev_ptr), &input.pitch, sizeof(uchar3) * width, height, 4);
  printf("cuMemAllocPitch() -> %u, pitch: %lu\n", err, input.pitch);

  input.width = width;
  input.height = height;

  CUDA_MEMCPY2D params = {0};

  params.srcXInBytes = 0;
  params.srcY = 0;

  params.srcMemoryType = CU_MEMORYTYPE_HOST;
  params.srcPitch = sizeof(uchar3) * width;

  params.dstXInBytes = 0;
  params.dstY = 0;

  params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  params.dstDevice = (CUdeviceptr) input.dev_ptr;
  params.dstPitch = input.pitch;

  params.WidthInBytes = sizeof(uchar3) * width;
  params.Height = height;

  uint32_t n_tags = 0;
  cuAprilTagsID_t tags[ALLOC_TAGS] = {};

  CUstream cuda_stream = {};
  err = cuStreamCreate(&(cuda_stream), CU_STREAM_DEFAULT);
  printf("cudaStreamCreate() -> %u\n", err);

  unsigned int total = 0;

  Instant start;

  for (int i = 0; i < ITERS; i++) {
  if (i == WARMUP) start = Clock::now();
  
  for (size_t i = 0; i < imgs.size(); i++) {
      params.srcHost = (void*) imgs[i].first.data;
      cuMemcpy2D(&params);
      
      cuAprilTagsDetect(handle, &input, (cuAprilTagsID_t*) &tags, &n_tags, ALLOC_TAGS, NULL);

      total += n_tags;
      n_tags = 0;
  }
  }
  
  Instant end = Clock::now();
  Duration elapsed = end - start;

  printf("cuAprilTagsDetect: %u detections in %f ms\n", total, elapsed.count());
  total = 0;

  // ================= OpenCV ArUco =================

  std::vector<int> markerIds;
  std::vector<std::vector<cv::Point2f>> markerCorners;
  
  cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
  cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11);
  
  cv::aruco::ArucoDetector detector(dictionary, detectorParams);

  for (int i = 0; i < ITERS; i++) {
  if (i == WARMUP) start = Clock::now();
  
  for (size_t i = 0; i < imgs.size(); i++) {
    detector.detectMarkers(imgs[i].first, markerCorners, markerIds);
    total += markerIds.size();
    markerIds.clear();
  }
  }
  
  end = Clock::now();
  elapsed = end - start;
  
  printf("cv::ArucoDetector::detectMarkers: %u detections in %f ms\n", total, elapsed.count());
  total = 0;

  // black and white variation


  for (int i = 0; i < ITERS; i++) {
  if (i == WARMUP) start = Clock::now();
  
  for (size_t i = 0; i < imgs.size(); i++) {
    detector.detectMarkers(imgs[i].second, markerCorners, markerIds);
    total += markerIds.size();
    markerIds.clear();
  }
  }
  
  end = Clock::now();
  elapsed = end - start;
  
  printf("cv::ArucoDetector::detectMarkers (black and white): %u detections in %f ms\n", total, elapsed.count());
  total = 0;
  
  // ================ AprilTag ================

  apriltag_detector_t *td = apriltag_detector_create();
  
  apriltag_family_t *tf = tag36h11_create();
  apriltag_detector_add_family(td, tf);

  image_u8_t im = {
    .width = width,
    .height = height,
    .stride = width
  };

  for (int i = 0; i < ITERS; i++) {
  if (i == WARMUP) start = Clock::now();
  
  for (size_t i = 0; i < imgs.size(); i++) {
    im.buf = imgs[i].second.data;
    zarray_t *detections = apriltag_detector_detect(td, &im);
    total += zarray_size(detections);
    apriltag_detections_destroy(detections);
  }
  }

  end = Clock::now();
  elapsed = end - start;

  printf("apriltag_detector_detect: %u detections in %f ms\n", total, elapsed.count());
  

  // ============================================
  
  cuStreamDestroy(cuda_stream);
  cuMemFree((CUdeviceptr) input.dev_ptr);
  cuAprilTagsDestroy(handle);

  tag36h11_destroy(tf);
  apriltag_detector_destroy(td);

  return 0;
}
