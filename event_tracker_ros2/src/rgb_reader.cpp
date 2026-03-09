// RGB frame reader — identical to standalone version.
// Only the include path has changed for ROS 2 package layout.

#include "event_tracker/rgb_reader.h"
#include <H5Cpp.h>
#include <iostream>
#include <iomanip>
#include <algorithm>

bool RGBReader::open(const std::string& path) {
    cap_.open(path);
    if (!cap_.isOpened()) {
        std::cerr << "RGBReader: cannot open " << path << "\n";
        return false;
    }

    width_  = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    height_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    fps_    = cap_.get(cv::CAP_PROP_FPS);
    total_frames_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_COUNT));
    duration_ = (fps_ > 0) ? total_frames_ / fps_ : 0;

    std::cout << "  RGB video : " << width_ << " x " << height_
              << "  " << fps_ << " fps  " << total_frames_ << " frames  "
              << std::fixed << std::setprecision(1) << duration_ << " s\n";

    prev_frame_idx_ = -1;
    prev_gray_ = cv::Mat();
    return true;
}

bool RGBReader::openH5(const std::string& h5_path) {
    h5_path_ = h5_path;
    h5_mode_ = false;

    try {
        H5::H5File file(h5_path, H5F_ACC_RDONLY);

        if (!file.nameExists("ovc") || !file.nameExists("ovc/rgb") ||
            !file.nameExists("ovc/rgb/data")) {
            return false;
        }

        {
            H5::DataSet ds = file.openDataSet("ovc/rgb/data");
            H5::DataSpace sp = ds.getSpace();
            int ndims = sp.getSimpleExtentNdims();
            if (ndims != 4) return false;

            hsize_t dims[4];
            sp.getSimpleExtentDims(dims);
            total_frames_ = static_cast<int>(dims[0]);
            height_ = static_cast<int>(dims[1]);
            width_ = static_cast<int>(dims[2]);
        }

        {
            H5::DataSet ds = file.openDataSet("ovc/ts");
            H5::DataSpace sp = ds.getSpace();
            hsize_t dims[1];
            sp.getSimpleExtentDims(dims);
            std::vector<int64_t> ts_us(dims[0]);
            ds.read(ts_us.data(), H5::PredType::NATIVE_INT64);

            int64_t evt_t0 = 0;
            if (file.nameExists("prophesee/left/t")) {
                H5::DataSet tds = file.openDataSet("prophesee/left/t");
                hsize_t one = 1, zero = 0;
                H5::DataSpace fsp = tds.getSpace();
                fsp.selectHyperslab(H5S_SELECT_SET, &one, &zero);
                H5::DataSpace msp(1, &one);
                tds.read(&evt_t0, H5::PredType::NATIVE_INT64, msp, fsp);
            }

            h5_timestamps_.resize(dims[0]);
            for (size_t i = 0; i < dims[0]; ++i) {
                h5_timestamps_[i] = (ts_us[i] - evt_t0) * 1e-6;
            }
        }

        file.close();
    } catch (H5::Exception& e) {
        std::cerr << "RGBReader H5 error: " << e.getDetailMsg() << "\n";
        return false;
    }

    if (h5_timestamps_.size() > 1) {
        double dt = h5_timestamps_.back() - h5_timestamps_.front();
        fps_ = (total_frames_ - 1) / dt;
    } else {
        fps_ = 25.0;
    }
    duration_ = h5_timestamps_.empty() ? 0.0
              : (h5_timestamps_.back() - h5_timestamps_.front());

    h5_mode_ = true;
    prev_frame_idx_ = -1;
    prev_gray_ = cv::Mat();

    std::cout << "  RGB (H5)  : " << width_ << " x " << height_
              << "  " << std::fixed << std::setprecision(1) << fps_ << " fps  "
              << total_frames_ << " frames  " << duration_ << " s\n";
    return true;
}

int RGBReader::findH5Frame(double t_sec) const {
    if (h5_timestamps_.empty()) return -1;
    auto it = std::lower_bound(h5_timestamps_.begin(), h5_timestamps_.end(), t_sec);
    if (it == h5_timestamps_.end()) return total_frames_ - 1;
    if (it == h5_timestamps_.begin()) return 0;
    auto prev_it = std::prev(it);
    if (std::abs(*it - t_sec) < std::abs(*prev_it - t_sec))
        return static_cast<int>(it - h5_timestamps_.begin());
    else
        return static_cast<int>(prev_it - h5_timestamps_.begin());
}

bool RGBReader::readH5FrameAt(int idx, cv::Mat& bgr, cv::Mat& gray) {
    if (idx < 0 || idx >= total_frames_) return false;

    if (idx == cached_frame_idx_ && !cached_bgr_.empty()) {
        bgr = cached_bgr_;
        gray = cached_gray_;
        return true;
    }

    try {
        H5::H5File file(h5_path_, H5F_ACC_RDONLY);
        H5::DataSet ds = file.openDataSet("ovc/rgb/data");
        H5::DataSpace fspace = ds.getSpace();

        hsize_t offset[4] = { (hsize_t)idx, 0, 0, 0 };
        hsize_t count[4] = { 1, (hsize_t)height_, (hsize_t)width_, 3 };
        fspace.selectHyperslab(H5S_SELECT_SET, count, offset);

        hsize_t mem_dims[3] = { (hsize_t)height_, (hsize_t)width_, 3 };
        H5::DataSpace mspace(3, mem_dims);

        cv::Mat rgb(height_, width_, CV_8UC3);
        ds.read(rgb.data, H5::PredType::NATIVE_UINT8, mspace, fspace);

        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

        file.close();
    } catch (H5::Exception& e) {
        std::cerr << "RGBReader H5 read error at frame " << idx << ": "
                  << e.getDetailMsg() << "\n";
        return false;
    }

    cached_bgr_ = bgr.clone();
    cached_gray_ = gray.clone();
    cached_frame_idx_ = idx;
    return true;
}

bool RGBReader::readFrameAt(int idx, cv::Mat& bgr, cv::Mat& gray) {
    if (h5_mode_)
        return readH5FrameAt(idx, bgr, gray);

    if (!cap_.isOpened() || idx < 0 || idx >= total_frames_) return false;

    if (idx == cached_frame_idx_ && !cached_bgr_.empty()) {
        bgr = cached_bgr_;
        gray = cached_gray_;
        return true;
    }

    int cur_pos = static_cast<int>(cap_.get(cv::CAP_PROP_POS_FRAMES));
    if (cur_pos != idx) {
        cap_.set(cv::CAP_PROP_POS_FRAMES, idx);
    }

    cv::Mat raw;
    if (!cap_.read(raw) || raw.empty()) return false;

    if (raw.channels() == 4)
        cv::cvtColor(raw, bgr, cv::COLOR_BGRA2BGR);
    else
        bgr = raw;

    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    cached_bgr_ = bgr.clone();
    cached_gray_ = gray.clone();
    cached_frame_idx_ = idx;

    return true;
}

bool RGBReader::getFrame(double t_sec, cv::Mat& frame) {
    if (h5_mode_) {
        int idx = findH5Frame(t_sec);
        cv::Mat gray;
        return readFrameAt(idx, frame, gray);
    }
    if (!cap_.isOpened() || fps_ <= 0) return false;
    int idx = static_cast<int>(t_sec * fps_);
    cv::Mat gray;
    return readFrameAt(idx, frame, gray);
}

void RGBReader::resizeToTarget(cv::Mat& mask) const {
    if (target_w_ > 0 && target_h_ > 0 &&
        (mask.cols != target_w_ || mask.rows != target_h_)) {
        cv::Mat resized;
        cv::resize(mask, resized, cv::Size(target_w_, target_h_), 0, 0, cv::INTER_NEAREST);
        mask = resized;
    }
}

bool RGBReader::getMotionMask(double t_sec,
                              const Eigen::Matrix<double, 6, 1>& bg_affine,
                              double cx, double cy, double sx, double sy,
                              cv::Mat& mask)
{
    if (!isOpen()) return false;

    int frame_idx = h5_mode_ ? findH5Frame(t_sec)
                             : static_cast<int>(t_sec * fps_);
    if (frame_idx < 1 || frame_idx >= total_frames_) return false;

    if (frame_idx == cached_mask_idx_ && !cached_mask_.empty()) {
        mask = cached_mask_;
        return true;
    }

    cv::Mat curr_bgr, curr_gray, prev_bgr, prev_gray;
    if (!readFrameAt(frame_idx, curr_bgr, curr_gray)) return false;
    if (!readFrameAt(frame_idx - 1, prev_bgr, prev_gray)) return false;

    if (target_w_ > 0 && target_h_ > 0) {
        cv::Size target(target_w_, target_h_);
        if (curr_gray.cols != target_w_ || curr_gray.rows != target_h_) {
            cv::resize(curr_gray, curr_gray, target);
            cv::resize(prev_gray, prev_gray, target);
        }
    }

    int W = curr_gray.cols;
    int H = curr_gray.rows;
    double dt = 1.0 / fps_;

    cv::Mat map_x(H, W, CV_32F);
    cv::Mat map_y(H, W, CV_32F);

    for (int y = 0; y < H; ++y) {
        float* mx = map_x.ptr<float>(y);
        float* my = map_y.ptr<float>(y);
        double yn = ((double)y - cy) / sy;
        for (int x = 0; x < W; ++x) {
            double xn = ((double)x - cx) / sx;
            double vx = bg_affine(0)*xn + bg_affine(1)*yn + bg_affine(4);
            double vy = bg_affine(2)*xn + bg_affine(3)*yn + bg_affine(5);
            mx[x] = static_cast<float>(x - vx * dt);
            my[x] = static_cast<float>(y - vy * dt);
        }
    }

    cv::Mat aligned_prev;
    cv::remap(prev_gray, aligned_prev, map_x, map_y, cv::INTER_LINEAR,
              cv::BORDER_REPLICATE);

    cv::Mat diff;
    cv::absdiff(curr_gray, aligned_prev, diff);

    cv::GaussianBlur(diff, diff, cv::Size(7, 7), 2.0);

    double otsu_thresh = cv::threshold(diff, mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    double effective_thresh = std::max(otsu_thresh, 20.0);
    cv::threshold(diff, mask, effective_thresh, 255, cv::THRESH_BINARY);

    cv::Mat k_close = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k_close);

    cv::Mat k_open = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, k_open);

    cv::Mat k_dilate = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
    cv::dilate(mask, mask, k_dilate);

    prev_gray_ = curr_gray.clone();
    prev_frame_idx_ = frame_idx;
    cached_mask_ = mask.clone();
    cached_mask_idx_ = frame_idx;

    return true;
}

bool RGBReader::getMotionMaskSimple(double t_sec, cv::Mat& mask) {
    if (!isOpen()) return false;

    int frame_idx = h5_mode_ ? findH5Frame(t_sec)
                             : static_cast<int>(t_sec * fps_);
    if (frame_idx < 1 || frame_idx >= total_frames_) return false;

    cv::Mat curr_bgr, curr_gray, prev_bgr, prev_gray;
    if (!readFrameAt(frame_idx, curr_bgr, curr_gray)) return false;
    if (!readFrameAt(frame_idx - 1, prev_bgr, prev_gray)) return false;

    if (target_w_ > 0 && target_h_ > 0) {
        cv::Size target(target_w_, target_h_);
        if (curr_gray.cols != target_w_ || curr_gray.rows != target_h_) {
            cv::resize(curr_gray, curr_gray, target);
            cv::resize(prev_gray, prev_gray, target);
        }
    }

    cv::Mat diff;
    cv::absdiff(curr_gray, prev_gray, diff);
    cv::GaussianBlur(diff, diff, cv::Size(5, 5), 2.0);
    cv::threshold(diff, mask, 15, 255, cv::THRESH_BINARY);

    cv::Mat k_close = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
    cv::Mat k_open  = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k_close);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  k_open);

    cv::Mat k_dilate = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(21, 21));
    cv::dilate(mask, mask, k_dilate);

    return true;
}
