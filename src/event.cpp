#include "event.h"
#include <H5Cpp.h>
#include <algorithm>
#include <iostream>
#include <cmath>

bool H5EventReader::open(const std::string& path, const std::string& side) {
    path_  = path;
    group_ = "prophesee/" + side;

    try {
        H5::H5File file(path, H5F_ACC_RDONLY);

        // --- resolution ---
        {
            H5::DataSet ds = file.openDataSet(group_ + "/calib/resolution");
            int64_t res[2];
            ds.read(res, H5::PredType::NATIVE_INT64);
            width_  = static_cast<uint16_t>(res[0]);
            height_ = static_cast<uint16_t>(res[1]);
        }

        // --- total event count (from the t dataset) ---
        {
            H5::DataSet ds = file.openDataSet(group_ + "/t");
            H5::DataSpace sp = ds.getSpace();
            hsize_t dims[1];
            sp.getSimpleExtentDims(dims);
            total_events_ = static_cast<size_t>(dims[0]);
        }

        // --- first timestamp (offset) ---
        {
            H5::DataSet ds = file.openDataSet(group_ + "/t");
            int64_t first;
            hsize_t offset = 0, count = 1;
            H5::DataSpace fspace = ds.getSpace();
            fspace.selectHyperslab(H5S_SELECT_SET, &count, &offset);
            H5::DataSpace mspace(1, &count);
            ds.read(&first, H5::PredType::NATIVE_INT64, mspace, fspace);
            t0_us_ = first;
        }

        // --- ms_map_idx ---
        {
            H5::DataSet ds = file.openDataSet(group_ + "/ms_map_idx");
            H5::DataSpace sp = ds.getSpace();
            hsize_t dims[1];
            sp.getSimpleExtentDims(dims);
            ms_map_idx_.resize(dims[0]);
            ds.read(ms_map_idx_.data(), H5::PredType::NATIVE_UINT64);
        }

        file.close();
    } catch (H5::Exception& e) {
        std::cerr << "HDF5 error opening " << path << ": "
                  << e.getDetailMsg() << "\n";
        return false;
    }

    std::cout << "Opened " << path << "  [" << group_ << "]\n"
              << "  Resolution : " << width_ << " x " << height_ << "\n"
              << "  Events     : " << total_events_ << "\n"
              << "  ms entries : " << ms_map_idx_.size() << "\n";
    return true;
}

double H5EventReader::totalDurationSec() const {
    if (ms_map_idx_.empty()) return 0.0;
    return ms_map_idx_.size() / 1000.0;   // one entry per millisecond
}

bool H5EventReader::readBatch(double t0_sec, double duration_sec,
                              EventBatch& batch) {
    // Use the ms_map_idx table to jump straight to the right events
    // instead of scanning through the whole array
    size_t ms_start = static_cast<size_t>(t0_sec * 1000.0);
    size_t ms_end   = static_cast<size_t>((t0_sec + duration_sec) * 1000.0);

    if (ms_start >= ms_map_idx_.size()) return false;
    ms_end = std::min(ms_end, ms_map_idx_.size());

    size_t idx_begin = ms_map_idx_[ms_start];
    size_t idx_end   = (ms_end < ms_map_idx_.size())
                         ? ms_map_idx_[ms_end]
                         : total_events_;

    return readRange(idx_begin, idx_end, batch);
}

bool H5EventReader::readRange(size_t idx_begin, size_t idx_end,
                              EventBatch& batch) {
    if (idx_begin >= idx_end || idx_begin >= total_events_) return false;
    idx_end = std::min(idx_end, total_events_);

    size_t n = idx_end - idx_begin;
    hsize_t offset = idx_begin;
    hsize_t count  = n;

    std::vector<int64_t>  t_buf(n);
    std::vector<uint16_t> x_buf(n);
    std::vector<uint16_t> y_buf(n);
    std::vector<int8_t>   p_buf(n);

    try {
        H5::H5File file(path_, H5F_ACC_RDONLY);
        H5::DataSpace memspace(1, &count);

        auto readDS = [&](const std::string& name, void* buf,
                          const H5::DataType& type) {
            H5::DataSet ds = file.openDataSet(group_ + "/" + name);
            H5::DataSpace fspace = ds.getSpace();
            fspace.selectHyperslab(H5S_SELECT_SET, &count, &offset);
            ds.read(buf, type, memspace, fspace);
        };

        readDS("t", t_buf.data(), H5::PredType::NATIVE_INT64);
        readDS("x", x_buf.data(), H5::PredType::NATIVE_UINT16);
        readDS("y", y_buf.data(), H5::PredType::NATIVE_UINT16);
        readDS("p", p_buf.data(), H5::PredType::NATIVE_INT8);

        file.close();
    } catch (H5::Exception& e) {
        std::cerr << "HDF5 read error: " << e.getDetailMsg() << "\n";
        return false;
    }

    // Pack into Event structs, converting timestamps from µs to seconds
    batch.events.resize(n);
    batch.width  = width_;
    batch.height = height_;

    for (size_t i = 0; i < n; ++i) {
        batch.events[i].t = (t_buf[i] - t0_us_) * 1e-6;   // µs → s
        batch.events[i].x = x_buf[i];
        batch.events[i].y = y_buf[i];
        batch.events[i].p = p_buf[i];
    }

    return true;
}
