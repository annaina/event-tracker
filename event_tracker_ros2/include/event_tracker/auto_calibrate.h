#ifndef EVENT_TRACKER__AUTO_CALIBRATE_H
#define EVENT_TRACKER__AUTO_CALIBRATE_H

// Auto-calibration: find the affine mapping between event and RGB cameras.
//
// Unchanged from the standalone project. Given an RGB frame and a batch of
// events from the same moment, we find the (sx, sy, ox, oy) transform that
// best aligns event edges with RGB edges.
//
// The transform is:
//   u_evt = sx * u_rgb + ox
//   v_evt = sy * v_rgb + oy

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <vector>
#include <fstream>
#include <string>

#include "event_tracker/event.h"

struct CalibResult {
    double sx, sy, ox, oy;
    double score;   // 0-1, fraction of event edges matching RGB edges
};

// Load a calibration file written by the calibrate tool.
inline bool loadCalibration(const std::string& path, CalibResult& out) {
    std::ifstream f(path);
    if (!f.is_open()) return false;

    out = {1.0, 1.0, 0.0, 0.0, 0.0};
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto pos = line.find(':');
        if (pos == std::string::npos) continue;
        std::string key = line.substr(0, pos);
        double val = std::stod(line.substr(pos + 1));
        if      (key == "sx")    out.sx    = val;
        else if (key == "sy")    out.sy    = val;
        else if (key == "ox")    out.ox    = val;
        else if (key == "oy")    out.oy    = val;
        else if (key == "score") out.score = val;
    }
    return true;
}

// Build a binary edge image from events.
inline cv::Mat buildEventEdges(const EventBatch& events, int w, int h) {
    cv::Mat img(h, w, CV_8UC1, cv::Scalar(0));
    for (const auto& e : events.events) {
        if (e.x >= 0 && e.x < w && e.y >= 0 && e.y < h)
            img.at<uchar>(e.y, e.x) = 255;
    }
    cv::GaussianBlur(img, img, cv::Size(3, 3), 0.8);
    cv::threshold(img, img, 30, 255, cv::THRESH_BINARY);
    return img;
}

// Score an alignment candidate.
inline double scoreAlignment(const cv::Mat& rgb_edges,
                             const cv::Mat& evt_edges,
                             double sx, double sy, double ox, double oy) {
    int rw = rgb_edges.cols, rh = rgb_edges.rows;
    int ew = evt_edges.cols, eh = evt_edges.rows;

    cv::Mat dilated;
    cv::dilate(rgb_edges, dilated, cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5}));

    long hits = 0, total = 0;
    for (int ey = 0; ey < eh; ++ey) {
        int ry = (int)std::round((ey - oy) / sy);
        if (ry < 0 || ry >= rh) continue;

        for (int ex = 0; ex < ew; ++ex) {
            if (evt_edges.at<uchar>(ey, ex) == 0) continue;
            total++;

            int rx = (int)std::round((ex - ox) / sx);
            if (rx < 0 || rx >= rw) continue;

            if (dilated.at<uchar>(ry, rx) > 0)
                hits++;
        }
    }
    return (total > 0) ? (double)hits / total : 0.0;
}

// Run auto-calibration: 3-pass grid search.
inline CalibResult autoCalibrate(
    const std::vector<cv::Mat>& rgb_edges_vec,
    const std::vector<cv::Mat>& evt_edges_vec,
    double init_sx, double init_sy,
    double init_ox, double init_oy,
    bool wide = false)
{
    auto t0 = std::chrono::steady_clock::now();
    int n_frames = (int)rgb_edges_vec.size();

    // Coarse pass: 4x downsample, 1 frame
    cv::Mat coarse_rgb, coarse_evt;
    cv::resize(rgb_edges_vec[0], coarse_rgb, cv::Size(), 0.25, 0.25, cv::INTER_NEAREST);
    cv::resize(evt_edges_vec[0], coarse_evt, cv::Size(), 0.25, 0.25, cv::INTER_NEAREST);

    auto coarseScore = [&](double sx, double sy, double ox, double oy) {
        return scoreAlignment(coarse_rgb, coarse_evt,
                              sx, sy, ox * 0.25, oy * 0.25);
    };

    // Fine pass: half-res, min(4, n_frames) frames
    int n_fine = std::min(4, n_frames);
    std::vector<cv::Mat> half_rgb(n_fine), half_evt(n_fine);
    for (int i = 0; i < n_fine; i++) {
        cv::resize(rgb_edges_vec[i], half_rgb[i], cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
        cv::resize(evt_edges_vec[i], half_evt[i], cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
    }
    auto fineScore = [&](double sx, double sy, double ox, double oy) {
        double total = 0;
        for (int i = 0; i < n_fine; i++)
            total += scoreAlignment(half_rgb[i], half_evt[i],
                                    sx, sy, ox * 0.5, oy * 0.5);
        return total / n_fine;
    };

    // Sub-pixel pass: full-res, all frames
    auto fullScore = [&](double sx, double sy, double ox, double oy) {
        double total = 0;
        for (int i = 0; i < n_frames; i++)
            total += scoreAlignment(rgb_edges_vec[i], evt_edges_vec[i], sx, sy, ox, oy);
        return total / n_frames;
    };

    CalibResult best{init_sx, init_sy, init_ox, init_oy, 0.0};

    double sx_range = wide ? 0.40 : 0.10;
    double sx_step  = wide ? 0.04 : 0.02;
    double ox_range = wide ? 150  : 50;
    double ox_step  = wide ? 15   : 10;

    // --- Pass 1: coarse ---
    std::cout << "  [AutoCal] Pass 1/3: coarse search"
              << (wide ? " (wide)" : "") << "...\n";
    double best_score = 0;
    for (double dsx = -sx_range; dsx <= sx_range; dsx += sx_step) {
        for (double dsy = -sx_range; dsy <= sx_range; dsy += sx_step) {
            for (double dox = -ox_range; dox <= ox_range; dox += ox_step) {
                for (double doy = -ox_range; doy <= ox_range; doy += ox_step) {
                    double sx = init_sx + dsx;
                    double sy = init_sy + dsy;
                    double ox = init_ox + dox;
                    double oy = init_oy + doy;
                    if (sx < 0.2 || sy < 0.2) continue;

                    double s = coarseScore(sx, sy, ox, oy);
                    if (s > best_score) {
                        best_score = s;
                        best = {sx, sy, ox, oy, s};
                    }
                }
            }
        }
    }
    std::cout << "    coarse: " << (int)(best.score*100) << "% match"
              << "  sx=" << best.sx << " sy=" << best.sy
              << " ox=" << best.ox << " oy=" << best.oy << "\n";

    // --- Pass 2: fine ---
    std::cout << "  [AutoCal] Pass 2/3: fine search (" << n_fine << " frames, half-res)...\n";
    CalibResult fine = best;
    double fine_score = 0;
    for (double dsx = -0.02; dsx <= 0.02; dsx += 0.005) {
        for (double dsy = -0.02; dsy <= 0.02; dsy += 0.005) {
            for (double dox = -12; dox <= 12; dox += 3) {
                for (double doy = -12; doy <= 12; doy += 3) {
                    double sx = best.sx + dsx;
                    double sy = best.sy + dsy;
                    double ox = best.ox + dox;
                    double oy = best.oy + doy;

                    double s = fineScore(sx, sy, ox, oy);
                    if (s > fine_score) {
                        fine_score = s;
                        fine = {sx, sy, ox, oy, s};
                    }
                }
            }
        }
    }
    std::cout << "    fine: " << (int)(fine.score*100) << "%"
              << "  sx=" << fine.sx << " sy=" << fine.sy
              << " ox=" << fine.ox << " oy=" << fine.oy << "\n";

    // --- Pass 3: sub-pixel ---
    std::cout << "  [AutoCal] Pass 3/3: sub-pixel (" << n_frames << " frames, full-res)...\n";
    CalibResult sub = fine;
    double sub_score = fine.score;
    for (double dsx = -0.003; dsx <= 0.003; dsx += 0.001) {
        for (double dsy = -0.003; dsy <= 0.003; dsy += 0.001) {
            for (double dox = -3; dox <= 3; dox += 1) {
                for (double doy = -3; doy <= 3; doy += 1) {
                    double sx = fine.sx + dsx;
                    double sy = fine.sy + dsy;
                    double ox = fine.ox + dox;
                    double oy = fine.oy + doy;

                    double s = fullScore(sx, sy, ox, oy);
                    if (s > sub_score) {
                        sub_score = s;
                        sub = {sx, sy, ox, oy, s};
                    }
                }
            }
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "  [AutoCal] Done in " << std::fixed << std::setprecision(1)
              << elapsed << "s — " << (int)(sub.score*100) << "% edge overlap\n"
              << "    sx=" << std::setprecision(4) << sub.sx
              << "  sy=" << sub.sy
              << "  ox=" << std::setprecision(1) << sub.ox
              << "  oy=" << sub.oy << "\n";

    return sub;
}

#endif // EVENT_TRACKER__AUTO_CALIBRATE_H
