#include <stdint.h>

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/pose_layers.hpp"

// caffe.proto > LayerParameter > WindowMulticlassKeypointDataParameter
//   'source' field specifies the window_file
//   'crop_size' indicates the desired warped size

#if CV_VERSION_MAJOR == 3
const int CV_LOAD_IMAGE_COLOR = cv::IMREAD_COLOR;
#endif

namespace caffe {

template <typename Dtype>
WindowMulticlassKeypointDataLayer<Dtype>::~WindowMulticlassKeypointDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void WindowMulticlassKeypointDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // LayerSetUp runs through the window_file and creates two structures
  // that hold windows: one for foreground (object) windows and one
  // for background (non-object) windows. We use an overlap threshold
  // to decide which is which.

  // window_file format
  // repeated:
  //    # image_index
  //    img_path (abs path)
  //    channels
  //    height
  //    width
  //    num_windows
  //    class_index overlap x1 y1 x2 y2 numkps classStart classEnd kp1 flipkp1 kp2 flipkp2 .. kpN flipkpN

  LOG(INFO) << "Window data layer:" << std::endl
      << "  foreground (object) overlap threshold: "
      << this->layer_param_.window_data_param().fg_threshold() << std::endl
      << "  background (non-object) overlap threshold: "
      << this->layer_param_.window_data_param().bg_threshold() << std::endl
      << "  foreground sampling fraction: "
      << this->layer_param_.window_data_param().fg_fraction() << std::endl
      << "  cache_images: "
      << this->layer_param_.window_data_param().cache_images() << std::endl
      << "  root_folder: "
      << this->layer_param_.window_data_param().root_folder();

  cache_images_ = this->layer_param_.window_data_param().cache_images();
  string root_folder = this->layer_param_.window_data_param().root_folder();

  const bool prefetch_needs_rand =
      this->transform_param_.mirror() ||
      this->transform_param_.crop_size();
  if (prefetch_needs_rand) {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  } else {
    prefetch_rng_.reset();
  }

  std::ifstream infile(this->layer_param_.window_data_param().source().c_str());
  CHECK(infile.good()) << "Failed to open window file "
      << this->layer_param_.window_data_param().source() << std::endl;

  map<int, int> label_hist;
  label_hist.insert(std::make_pair(0, 0));
  int numTotKps;

  string hashtag;
  int image_index, channels;
  if (!(infile >> hashtag >> image_index)) {
    LOG(FATAL) << "Window file is empty";
  }
  do {
    CHECK_EQ(hashtag, "#");
    // read image path
    string image_path;
    infile >> image_path;
    image_path = root_folder + image_path;
    // read image dimensions
    vector<int> image_size(3);
    infile >> image_size[0] >> image_size[1] >> image_size[2] >> numTotKps;
    channels = image_size[0];
    image_database_.push_back(std::make_pair(image_path, image_size));

    if (cache_images_) {
      Datum datum;
      if (!ReadFileToDatum(image_path, &datum)) {
        LOG(ERROR) << "Could not open or find file " << image_path;
        return;
      }
      image_database_cache_.push_back(std::make_pair(image_path, datum));
    }
    // read each box
    int num_windows;
    infile >> num_windows;
    const float fg_threshold =
        this->layer_param_.window_data_param().fg_threshold();
    const float bg_threshold =
        this->layer_param_.window_data_param().bg_threshold();
    for (int i = 0; i < num_windows; ++i) {
      int label, x1, y1, x2, y2, numKps, classStart, classEnd;
      float overlap;
      infile >> label >> overlap >> x1 >> y1 >> x2 >> y2 >> numKps >> classStart >> classEnd;
      vector<int> window_kps(numKps);
      vector<float> window_kpsVals(numKps);
      vector<int> window_flipkps(numKps);
      for (int j=0;j<numKps;j++){
        infile >> window_kps[j] >> window_flipkps[j] >> window_kpsVals[j];
      }
      vector<float> window(WindowMulticlassKeypointDataLayer::NUM);
      window[WindowMulticlassKeypointDataLayer::IMAGE_INDEX] = image_index;
      window[WindowMulticlassKeypointDataLayer::LABEL] = label;
      window[WindowMulticlassKeypointDataLayer::OVERLAP] = overlap;
      window[WindowMulticlassKeypointDataLayer::X1] = x1;
      window[WindowMulticlassKeypointDataLayer::Y1] = y1;
      window[WindowMulticlassKeypointDataLayer::X2] = x2;
      window[WindowMulticlassKeypointDataLayer::Y2] = y2;
      window[WindowMulticlassKeypointDataLayer::NUMKPS] = numKps;
      window[WindowMulticlassKeypointDataLayer::CSTART] = classStart;
      window[WindowMulticlassKeypointDataLayer::CEND] = classEnd;
      window[WindowMulticlassKeypointDataLayer::TOTKPS] = numTotKps;

      // add window to foreground list or background list
      if (overlap >= fg_threshold) {
        int label = window[WindowMulticlassKeypointDataLayer::LABEL];
        CHECK_GT(label, 0);
        fg_windows_.push_back(window);
        fg_windows_values_.push_back(window_kpsVals);
        fg_windows_kps_.push_back(window_kps);
        fg_windows_flipkps_.push_back(window_flipkps);
        label_hist.insert(std::make_pair(label, 0));
        label_hist[label]++;
      } else if (overlap < bg_threshold) {
        // background window, force label and overlap to 0
        window[WindowMulticlassKeypointDataLayer::LABEL] = 0;
        window[WindowMulticlassKeypointDataLayer::OVERLAP] = 0;
        bg_windows_.push_back(window);
        bg_windows_kps_.push_back(window_kps);
        bg_windows_flipkps_.push_back(window_flipkps);
        bg_windows_values_.push_back(window_kpsVals);
        label_hist[0]++;
      }
    }

    if (image_index % 100 == 0) {
      LOG(INFO) << "num: " << image_index << " "
          << image_path << " "
          << image_size[0] << " "
          << image_size[1] << " "
          << image_size[2] << " "
          << "windows to process: " << num_windows;
    }
  } while (infile >> hashtag >> image_index);

  LOG(INFO) << "Number of images: " << image_index+1;

  for (map<int, int>::iterator it = label_hist.begin();
      it != label_hist.end(); ++it) {
    LOG(INFO) << "class " << it->first << " has " << label_hist[it->first]
              << " samples";
  }

  LOG(INFO) << "Amount of context padding: "
      << this->layer_param_.window_data_param().context_pad();

  LOG(INFO) << "Crop mode: "
      << this->layer_param_.window_data_param().crop_mode();

  // image
  const int crop_size = this->transform_param_.crop_size();
  CHECK_GT(crop_size, 0);
  const int batch_size = this->layer_param_.window_data_param().batch_size();
  top[0]->Reshape(batch_size, channels, crop_size, crop_size);
  this->prefetch_data_.Reshape(batch_size, channels, crop_size, crop_size);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  top[1]->Reshape(batch_size, numTotKps, 1, 1);
  this->prefetch_label_.Reshape(batch_size, numTotKps, 1, 1);

  top[2]->Reshape(batch_size, numTotKps, 1, 1);
  this->prefetch_filter_.Reshape(batch_size, numTotKps, 1, 1);

  // data mean
  has_mean_file_ = this->transform_param_.has_mean_file();
  has_mean_values_ = this->transform_param_.mean_value_size() > 0;
  if (has_mean_file_) {
    const string& mean_file =
          this->transform_param_.mean_file();
    LOG(INFO) << "Loading mean file from" << mean_file;
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  float meanVals[3] = {102.98,115.95,122.77};
  if (1) {
    CHECK(has_mean_file_ == false) <<
      "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < 3; ++c) {
      mean_values_.push_back(meanVals[c]);
    }
    CHECK(mean_values_.size() == 1 || mean_values_.size() == channels) <<
     "Specify either 1 mean_value or as many as channels: " << channels;
    if (channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }
}

template <typename Dtype>
unsigned int WindowMulticlassKeypointDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

// Thread fetching the data
template <typename Dtype>
void WindowMulticlassKeypointDataLayer<Dtype>::InternalThreadEntry() {
  // At each iteration, sample N windows where N*p are foreground (object)
  // windows and N*(1-p) are background (non-object) windows
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  Dtype* top_filter = this->prefetch_filter_.mutable_cpu_data();

  const Dtype scale = this->layer_param_.window_data_param().scale();
  const int batch_size = this->layer_param_.window_data_param().batch_size();
  const int context_pad = this->layer_param_.window_data_param().context_pad();
  const int crop_size = this->transform_param_.crop_size();
  const bool mirror = this->transform_param_.mirror();
  const float fg_fraction =
      this->layer_param_.window_data_param().fg_fraction();
  Dtype* mean = NULL;
  int mean_off = 0;
  int mean_width = 0;
  int mean_height = 0;
  if (this->has_mean_file_) {
    mean = this->data_mean_.mutable_cpu_data();
    mean_off = (this->data_mean_.width() - crop_size) / 2;
    mean_width = this->data_mean_.width();
    mean_height = this->data_mean_.height();
  }
  cv::Size cv_crop_size(crop_size, crop_size);
  const string& crop_mode = this->layer_param_.window_data_param().crop_mode();

  bool use_square = (crop_mode == "square") ? true : false;

  // zero out batch
  caffe_set(this->prefetch_data_.count(), Dtype(0), top_data);
  caffe_set(this->prefetch_label_.count(), Dtype(0), top_label);
  caffe_set(this->prefetch_filter_.count(), Dtype(0), top_filter);

  const int num_fg = static_cast<int>(static_cast<float>(batch_size)
      * fg_fraction);
  const int num_samples[2] = { batch_size - num_fg, num_fg };

  int item_id = 0;
  // sample from bg set then fg set
  for (int is_fg = 0; is_fg < 2; ++is_fg) {
    for (int dummy = 0; dummy < num_samples[is_fg]; ++dummy) {
      // sample a window
      timer.Start();
      const unsigned int rand_index = PrefetchRand();
      vector<float> window = (is_fg) ?
          fg_windows_[rand_index % fg_windows_.size()] :
          bg_windows_[rand_index % bg_windows_.size()];

      vector<int> window_kps = (is_fg) ?
        fg_windows_kps_[rand_index % fg_windows_.size()] :
        bg_windows_kps_[rand_index % bg_windows_.size()];

      vector<float> window_kpsVals = (is_fg) ?
            fg_windows_values_[rand_index % fg_windows_.size()] :
            bg_windows_values_[rand_index % bg_windows_.size()];

      bool do_mirror = mirror && PrefetchRand() % 2;

      if(do_mirror){
        window_kps = (is_fg) ?
            fg_windows_flipkps_[rand_index % fg_windows_.size()]:
            bg_windows_flipkps_[rand_index % bg_windows_.size()];
      }

      // load the image containing the window
      pair<std::string, vector<int> > image =
          image_database_[window[WindowMulticlassKeypointDataLayer<Dtype>::IMAGE_INDEX]];

      cv::Mat cv_img;
      cv_img = cv::imread(image.first, CV_LOAD_IMAGE_COLOR);
      if (!cv_img.data) {
          LOG(ERROR) << "Could not open or find file " << image.first;
          return;
      }
      read_time += timer.MicroSeconds();
      timer.Start();
      const int channels = cv_img.channels();

      // crop window out of image and warp it
      int x1 = window[WindowMulticlassKeypointDataLayer<Dtype>::X1];
      int y1 = window[WindowMulticlassKeypointDataLayer<Dtype>::Y1];
      int x2 = window[WindowMulticlassKeypointDataLayer<Dtype>::X2];
      int y2 = window[WindowMulticlassKeypointDataLayer<Dtype>::Y2];
      int numKps = window[WindowMulticlassKeypointDataLayer<Dtype>::NUMKPS];
      int classStart = window[WindowMulticlassKeypointDataLayer<Dtype>::CSTART];
      int classEnd = window[WindowMulticlassKeypointDataLayer<Dtype>::CEND];
      int numTotKps = window[WindowMulticlassKeypointDataLayer<Dtype>::TOTKPS];

      int pad_w = 0;
      int pad_h = 0;
      if (context_pad > 0 || use_square) {
        // scale factor by which to expand the original region
        // such that after warping the expanded region to crop_size x crop_size
        // there's exactly context_pad amount of padding on each side
        Dtype context_scale = static_cast<Dtype>(crop_size) /
            static_cast<Dtype>(crop_size - 2*context_pad);

        // compute the expanded region
        Dtype half_height = static_cast<Dtype>(y2-y1+1)/2.0;
        Dtype half_width = static_cast<Dtype>(x2-x1+1)/2.0;
        Dtype center_x = static_cast<Dtype>(x1) + half_width;
        Dtype center_y = static_cast<Dtype>(y1) + half_height;
        if (use_square) {
          if (half_height > half_width) {
            half_width = half_height;
          } else {
            half_height = half_width;
          }
        }
        x1 = static_cast<int>(round(center_x - half_width*context_scale));
        x2 = static_cast<int>(round(center_x + half_width*context_scale));
        y1 = static_cast<int>(round(center_y - half_height*context_scale));
        y2 = static_cast<int>(round(center_y + half_height*context_scale));

        // the expanded region may go outside of the image
        // so we compute the clipped (expanded) region and keep track of
        // the extent beyond the image
        int unclipped_height = y2-y1+1;
        int unclipped_width = x2-x1+1;
        int pad_x1 = std::max(0, -x1);
        int pad_y1 = std::max(0, -y1);
        int pad_x2 = std::max(0, x2 - cv_img.cols + 1);
        int pad_y2 = std::max(0, y2 - cv_img.rows + 1);
        // clip bounds
        x1 = x1 + pad_x1;
        x2 = x2 - pad_x2;
        y1 = y1 + pad_y1;
        y2 = y2 - pad_y2;
        CHECK_GT(x1, -1);
        CHECK_GT(y1, -1);
        CHECK_LT(x2, cv_img.cols);
        CHECK_LT(y2, cv_img.rows);

        int clipped_height = y2-y1+1;
        int clipped_width = x2-x1+1;

        // scale factors that would be used to warp the unclipped
        // expanded region
        Dtype scale_x =
            static_cast<Dtype>(crop_size)/static_cast<Dtype>(unclipped_width);
        Dtype scale_y =
            static_cast<Dtype>(crop_size)/static_cast<Dtype>(unclipped_height);

        // size to warp the clipped expanded region to
        cv_crop_size.width =
            static_cast<int>(round(static_cast<Dtype>(clipped_width)*scale_x));
        cv_crop_size.height =
            static_cast<int>(round(static_cast<Dtype>(clipped_height)*scale_y));
        pad_x1 = static_cast<int>(round(static_cast<Dtype>(pad_x1)*scale_x));
        pad_x2 = static_cast<int>(round(static_cast<Dtype>(pad_x2)*scale_x));
        pad_y1 = static_cast<int>(round(static_cast<Dtype>(pad_y1)*scale_y));
        pad_y2 = static_cast<int>(round(static_cast<Dtype>(pad_y2)*scale_y));

        pad_h = pad_y1;
        // if we're mirroring, we mirror the padding too (to be pedantic)
        if (do_mirror) {
          pad_w = pad_x2;
        } else {
          pad_w = pad_x1;
        }

        // ensure that the warped, clipped region plus the padding fits in the
        // crop_size x crop_size image (it might not due to rounding)
        if (pad_h + cv_crop_size.height > crop_size) {
          cv_crop_size.height = crop_size - pad_h;
        }
        if (pad_w + cv_crop_size.width > crop_size) {
          cv_crop_size.width = crop_size - pad_w;
        }
      }

      cv::Rect roi(x1, y1, x2-x1+1, y2-y1+1);
      cv::Mat cv_cropped_img = cv_img(roi);
      cv::resize(cv_cropped_img, cv_cropped_img,
          cv_crop_size, 0, 0, cv::INTER_LINEAR);

      // horizontal flip at random
      if (do_mirror) {
        cv::flip(cv_cropped_img, cv_cropped_img, 1);
      }

      // copy the warped window into top_data
      for (int h = 0; h < cv_cropped_img.rows; ++h) {
        const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
        int img_index = 0;
        for (int w = 0; w < cv_cropped_img.cols; ++w) {
          for (int c = 0; c < channels; ++c) {
            int top_index = ((item_id * channels + c) * crop_size + h + pad_h)
                     * crop_size + w + pad_w;
            // int top_index = (c * height + h) * width + w;
            Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
            if (this->has_mean_file_) {
              int mean_index = (c * mean_height + h + mean_off + pad_h)
                           * mean_width + w + mean_off + pad_w;
              top_data[top_index] = (pixel - mean[mean_index]) * scale;
            } else {
              if (1) {
                top_data[top_index] = (pixel - this->mean_values_[c]) * scale;
              } else {
                top_data[top_index] = pixel * scale;
              }
            }
          }
        }
      }
      trans_time += timer.MicroSeconds();
      // get window label
      for (int j=0;j<numKps;j++){
        top_label[item_id*numTotKps + window_kps[j]]=Dtype(window_kpsVals[j]);
      }
      for (int j=classStart;j<=classEnd;j++){
            top_filter[item_id*numTotKps + j]=Dtype(1);
      }

      #if 0
      // useful debugging code for dumping transformed windows to disk
      string file_id;
      std::stringstream ss;
      ss << PrefetchRand();
      ss >> file_id;
      std::ofstream inf((string("dump/") + file_id +
          string("_info.txt")).c_str(), std::ofstream::out);
      inf << image.first << std::endl
          << window[WindowMulticlassKeypointDataLayer<Dtype>::X1]+1 << std::endl
          << window[WindowMulticlassKeypointDataLayer<Dtype>::Y1]+1 << std::endl
          << window[WindowMulticlassKeypointDataLayer<Dtype>::X2]+1 << std::endl
          << window[WindowMulticlassKeypointDataLayer<Dtype>::Y2]+1 << std::endl
          << do_mirror << std::endl
          << top_label[item_id] << std::endl
          << is_fg << std::endl;
      inf.close();
      std::ofstream top_data_file((string("dump/") + file_id +
          string("_data.txt")).c_str(),
          std::ofstream::out | std::ofstream::binary);
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            top_data_file.write(reinterpret_cast<char*>(
                &top_data[((item_id * channels + c) * crop_size + h)
                          * crop_size + w]),
                sizeof(Dtype));
          }
        }
      }
      top_data_file.close();
      #endif

      item_id++;
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template <typename Dtype>
void WindowMulticlassKeypointDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  this->JoinPrefetchThread();
  DLOG(INFO) << "Thread joined";
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
            top[0]->mutable_cpu_data());
  caffe_copy(this->prefetch_label_.count(), this->prefetch_label_.cpu_data(),
            top[1]->mutable_cpu_data());
  caffe_copy(this->prefetch_filter_.count(), this->prefetch_filter_.cpu_data(),
            top[2]->mutable_cpu_data());
  float sumFilt = 0;
  //for (int i=0;i<this->prefetch_filter_.count();++i){
  //    sumFilt+=this->prefetch_filter_.cpu_data()[i];
  //}
  //LOG(INFO)<<sumFilt<<'\n';
  // Start a new prefetch thread
  DLOG(INFO) << "Prefetch copied";
  DLOG(INFO) << "CreatePrefetchThread";
  this->CreatePrefetchThread();
}

INSTANTIATE_CLASS(WindowMulticlassKeypointDataLayer);
REGISTER_LAYER_CLASS(WindowMulticlassKeypointData);
}  // namespace caffe
