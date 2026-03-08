// Camera Auto-Alignment Tool
//
// Finds the affine mapping (scale + offset) between the Prophesee event
// camera and the OVC RGB camera. We do this by maximizing edge overlap:
//
//   1. Run Canny on an RGB frame to get edges
//   2. Accumulate events into a binary image (also edges, since events
//      fire on brightness changes)
//   3. Warp the event image into RGB space with a candidate transform
//   4. Count how many event-edge pixels land on or near an RGB edge
//   5. Grid-search over (sx, sy, ox, oy) in three stages:
//      coarse → fine → sub-pixel
//
// The transform is just: u_evt = sx * u_rgb + ox
//                        v_evt = sy * v_rgb + oy
//
// Controls:
//   A      auto-align (single-frame grid search)
//   M      multi-frame auto-align (averages 5 frames, more robust)
//   S      print current parameters (ready to paste into tracker code)
//   N/P    next/previous frame
//   Sliders manual fine-tuning
//   Q/ESC  quit

#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>
#include <vector>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "rgb_reader.h"
#include "event.h"

// ---------------------------------------------------------------------------
//  Command-line arguments
// ---------------------------------------------------------------------------

struct Args {
    std::string h5path;
    std::string side    = "left";
    double      timeSec = 10.0;   // which moment in the recording to use
};

static Args parseArgs(int argc, char** argv) {
    Args a;
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <data.h5> [--side left|right] [--time S]\n";
        std::exit(1);
    }
    a.h5path = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--side" && i + 1 < argc)  a.side = argv[++i];
        else if (arg == "--time" && i + 1 < argc) a.timeSec = std::stod(argv[++i]);
    }
    return a;
}

// ---------------------------------------------------------------------------
//  Build an edge image from events
//
//  Since events fire at brightness changes, accumulating them over a
//  short window produces something very similar to an edge map. We blur
//  slightly to thicken the single-pixel events so they overlap better
//  with the Canny edges from the RGB side.
// ---------------------------------------------------------------------------

static cv::Mat buildEventEdgeImage(const EventBatch& events, int w, int h) {
    cv::Mat img(h, w, CV_8UC1, cv::Scalar(0));
    for (const auto& e : events.events) {
        if (e.x < w && e.y < h)
            img.at<uchar>(e.y, e.x) = 255;
    }
    cv::GaussianBlur(img, img, cv::Size(3, 3), 0.8);
    cv::threshold(img, img, 30, 255, cv::THRESH_BINARY);
    return img;
}

// ---------------------------------------------------------------------------
//  Score an alignment by warping event edges into RGB space
//
//  Forward mapping (RGB→Evt):  u_evt = sx*u_rgb + ox
//  Inverse (Evt→RGB):          u_rgb = (u_evt - ox) / sx
//
//  We dilate the RGB edges by a few pixels so near-misses still count.
//  The score is just: (# event pixels that land on a dilated RGB edge)
//                     / (total event edge pixels).
// ---------------------------------------------------------------------------

static double scoreAlignment(const cv::Mat& rgb_edges,
                             const cv::Mat& evt_edge_img,
                             double sx, double sy, double ox, double oy,
                             cv::Mat* overlay_out = nullptr)
{
    int rw = rgb_edges.cols, rh = rgb_edges.rows;
    int ew = evt_edge_img.cols, eh = evt_edge_img.rows;

    // Dilate RGB edges so we're not demanding pixel-perfect alignment
    cv::Mat rgb_dilated;
    cv::dilate(rgb_edges, rgb_dilated, cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5}));

    long hits = 0, total_evt = 0;
    cv::Mat overlay;
    if (overlay_out)
        overlay = cv::Mat(rh, rw, CV_8UC3, cv::Scalar(0));

    for (int ey = 0; ey < eh; ++ey) {
        int ry = (int)std::round((ey - oy) / sy);
        if (ry < 0 || ry >= rh) continue;

        for (int ex = 0; ex < ew; ++ex) {
            if (evt_edge_img.at<uchar>(ey, ex) == 0) continue;
            total_evt++;

            int rx = (int)std::round((ex - ox) / sx);
            if (rx < 0 || rx >= rw) continue;

            if (rgb_dilated.at<uchar>(ry, rx) > 0) {
                hits++;
                if (overlay_out)
                    overlay.at<cv::Vec3b>(ry, rx) = {0, 255, 0};   // green = match
            } else {
                if (overlay_out)
                    overlay.at<cv::Vec3b>(ry, rx) = {0, 0, 180};   // red = miss
            }
        }
    }

    if (overlay_out) *overlay_out = overlay;
    return (total_evt > 0) ? (double)hits / total_evt : 0.0;
}

// ---------------------------------------------------------------------------
//  Grid search for best alignment
//
//  Three passes, each narrowing around the previous best:
//    1. Coarse: ±10% scale, ±50px offset, step 2%/10px
//    2. Fine:   ±2% scale, ±12px offset, step 0.5%/2px
//    3. Sub-pixel: ±0.3% scale, ±3px, step 0.1%/1px
// ---------------------------------------------------------------------------

struct AlignParams {
    double sx, sy, ox, oy;
    double score;
};

static AlignParams gridSearch(const cv::Mat& rgb_edges,
                              const cv::Mat& evt_edge_img,
                              double sx_init, double sy_init,
                              double ox_init, double oy_init)
{
    std::cout << "\n  Auto-alignment: coarse grid search...\n";
    auto t0 = std::chrono::steady_clock::now();

    AlignParams best{sx_init, sy_init, ox_init, oy_init, 0.0};

    // --- Pass 1: coarse ---
    double best_score = 0;
    for (double dsx = -0.10; dsx <= 0.10; dsx += 0.02) {
        for (double dsy = -0.10; dsy <= 0.10; dsy += 0.02) {
            for (double dox = -50; dox <= 50; dox += 10) {
                for (double doy = -50; doy <= 50; doy += 10) {
                    double sx = sx_init + dsx;
                    double sy = sy_init + dsy;
                    double ox = ox_init + dox;
                    double oy = oy_init + doy;
                    if (sx < 0.3 || sy < 0.3) continue;

                    double s = scoreAlignment(rgb_edges, evt_edge_img, sx, sy, ox, oy);
                    if (s > best_score) {
                        best_score = s;
                        best = {sx, sy, ox, oy, s};
                    }
                }
            }
        }
    }

    std::cout << "  Coarse best: score=" << best.score
              << " sx=" << best.sx << " sy=" << best.sy
              << " ox=" << best.ox << " oy=" << best.oy << "\n";

    // --- Pass 2: fine ---
    std::cout << "  Refining...\n";
    double fine_best_score = best.score;
    AlignParams fine_best = best;
    for (double dsx = -0.02; dsx <= 0.02; dsx += 0.005) {
        for (double dsy = -0.02; dsy <= 0.02; dsy += 0.005) {
            for (double dox = -12; dox <= 12; dox += 2) {
                for (double doy = -12; doy <= 12; doy += 2) {
                    double sx = best.sx + dsx;
                    double sy = best.sy + dsy;
                    double ox = best.ox + dox;
                    double oy = best.oy + doy;

                    double s = scoreAlignment(rgb_edges, evt_edge_img, sx, sy, ox, oy);
                    if (s > fine_best_score) {
                        fine_best_score = s;
                        fine_best = {sx, sy, ox, oy, s};
                    }
                }
            }
        }
    }

    // --- Pass 3: sub-pixel ---
    std::cout << "  Sub-pixel refinement...\n";
    AlignParams sub_best = fine_best;
    double sub_best_score = fine_best.score;
    for (double dsx = -0.003; dsx <= 0.003; dsx += 0.001) {
        for (double dsy = -0.003; dsy <= 0.003; dsy += 0.001) {
            for (double dox = -3; dox <= 3; dox += 1) {
                for (double doy = -3; doy <= 3; doy += 1) {
                    double sx = fine_best.sx + dsx;
                    double sy = fine_best.sy + dsy;
                    double ox = fine_best.ox + dox;
                    double oy = fine_best.oy + doy;

                    double s = scoreAlignment(rgb_edges, evt_edge_img, sx, sy, ox, oy);
                    if (s > sub_best_score) {
                        sub_best_score = s;
                        sub_best = {sx, sy, ox, oy, s};
                    }
                }
            }
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "  Done in " << std::fixed << std::setprecision(1)
              << elapsed << "s\n";
    std::cout << "  Final: score=" << std::setprecision(4) << sub_best.score
              << "  sx=" << std::setprecision(4) << sub_best.sx
              << "  sy=" << sub_best.sy
              << "  ox=" << std::setprecision(1) << sub_best.ox
              << "  oy=" << sub_best.oy << "\n\n";

    return sub_best;
}

// ---------------------------------------------------------------------------
//  Build a composite visualization showing the edge overlap
//
//  The display is: dimmed RGB frame with RGB edges in blue, matching
//  event edges in green, and unmatched event edges in red. If the
//  alignment is good, you should see mostly green.
// ---------------------------------------------------------------------------

static cv::Mat buildComposite(const cv::Mat& rgb_frame,
                              const cv::Mat& rgb_edges,
                              const cv::Mat& evt_edge_img,
                              double sx, double sy, double ox, double oy,
                              double& score_out)
{
    cv::Mat overlay;
    score_out = scoreAlignment(rgb_edges, evt_edge_img, sx, sy, ox, oy, &overlay);

    cv::Mat composite;
    rgb_frame.copyTo(composite);
    composite *= 0.5;    // darken so edges pop

    // RGB edges in blue
    for (int y = 0; y < composite.rows; ++y)
        for (int x = 0; x < composite.cols; ++x)
            if (rgb_edges.at<uchar>(y, x) > 0)
                composite.at<cv::Vec3b>(y, x) = {255, 100, 0};

    // Event edges: green where they match, red where they don't
    for (int y = 0; y < composite.rows; ++y)
        for (int x = 0; x < composite.cols; ++x) {
            auto& ov = overlay.at<cv::Vec3b>(y, x);
            if (ov[1] > 0 || ov[2] > 0)
                composite.at<cv::Vec3b>(y, x) = ov;
        }

    // HUD text
    char buf[256];
    snprintf(buf, sizeof(buf), "Score: %.1f%%  sx=%.4f sy=%.4f ox=%.1f oy=%.1f",
             score_out * 100, sx, sy, ox, oy);
    cv::putText(composite, buf, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.65,
                {0, 255, 255}, 2, cv::LINE_AA);
    cv::putText(composite, "Blue=RGB edges  Green=matched events  Red=unmatched events",
                {10, composite.rows - 15}, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                {180, 180, 180}, 1, cv::LINE_AA);

    return composite;
}

// ---------------------------------------------------------------------------
//  Slider state
//
//  OpenCV trackbars only work with integers, so we map 0-1000 to the
//  actual float ranges we care about.
// ---------------------------------------------------------------------------

struct SliderState {
    int sx_slider = 500, sy_slider = 500;
    int ox_slider = 500, oy_slider = 500;

    // slider 0–1000 → scale 0.5–1.2
    double sx() const { return 0.5 + (sx_slider / 1000.0) * 0.7; }
    double sy() const { return 0.5 + (sy_slider / 1000.0) * 0.7; }

    // slider 0–1000 → offset -200 to +300
    double ox() const { return -200.0 + (ox_slider / 1000.0) * 500.0; }
    double oy() const { return -200.0 + (oy_slider / 1000.0) * 500.0; }

    void setFrom(double sx_, double sy_, double ox_, double oy_) {
        sx_slider = std::clamp((int)((sx_ - 0.5) / 0.7 * 1000), 0, 1000);
        sy_slider = std::clamp((int)((sy_ - 0.5) / 0.7 * 1000), 0, 1000);
        ox_slider = std::clamp((int)((ox_ + 200.0) / 500.0 * 1000), 0, 1000);
        oy_slider = std::clamp((int)((oy_ + 200.0) / 500.0 * 1000), 0, 1000);
    }
};

static SliderState g_sliders;
static bool g_slider_changed = true;
static void onSlider(int, void*) { g_slider_changed = true; }

// ===========================================================================
//  main
// ===========================================================================

int main(int argc, char** argv)
{
    Args args = parseArgs(argc, argv);

    // Open data
    RGBReader reader;
    if (!reader.openH5(args.h5path)) {
        std::cerr << "Failed to open H5 for RGB\n";
        return 1;
    }
    H5EventReader eventReader;
    if (!eventReader.open(args.h5path, args.side)) {
        std::cerr << "Failed to open H5 for events\n";
        return 1;
    }

    int evt_w = eventReader.width(), evt_h = eventReader.height();
    int rgb_w = reader.width(), rgb_h = reader.height();

    std::cout << "\n======================================================\n"
              << "  Camera Auto-Alignment Tool\n"
              << "  RGB : " << rgb_w << "x" << rgb_h << "  "
              << reader.numFrames() << " frames\n"
              << "  Evt : " << evt_w << "x" << evt_h << "  "
              << eventReader.totalEvents() << " events\n"
              << "======================================================\n"
              << "  A=auto-align  M=multi-frame  S=save/print\n"
              << "  N/P=next/prev frame  Sliders=fine-tune  Q=quit\n"
              << "======================================================\n\n";

    // Initial guess from the H5 camera intrinsics
    double fx_e = 1034.86, fy_e = 1033.48, cx_e = 629.70, cy_e = 357.60;
    double fx_r = 1268.56, fy_r = 1267.35, cx_r = 649.37, cy_r = 359.94;
    double init_sx = fx_e / fx_r;
    double init_sy = fy_e / fy_r;
    double init_ox = cx_e - cx_r * init_sx;
    double init_oy = cy_e - cy_r * init_sy;

    std::cout << "  Initial guess (from intrinsics):\n"
              << "    sx=" << init_sx << " sy=" << init_sy
              << " ox=" << init_ox << " oy=" << init_oy << "\n\n";

    g_sliders.setFrom(init_sx, init_sy, init_ox, init_oy);

    // Window + sliders
    const std::string win = "Camera Alignment";
    cv::namedWindow(win, cv::WINDOW_NORMAL);
    cv::resizeWindow(win, 1280, 800);

    cv::createTrackbar("Scale X", win, &g_sliders.sx_slider, 1000, onSlider);
    cv::createTrackbar("Scale Y", win, &g_sliders.sy_slider, 1000, onSlider);
    cv::createTrackbar("Offset X", win, &g_sliders.ox_slider, 1000, onSlider);
    cv::createTrackbar("Offset Y", win, &g_sliders.oy_slider, 1000, onSlider);

    // Load the first frame + corresponding events
    double t0 = reader.startTime();
    int frameIdx = std::min((int)(args.timeSec * reader.fps()), reader.numFrames() - 1);
    double evt_window = 0.015;   // 15ms of events — enough to see edges

    auto loadFrame = [&](cv::Mat& rgb_frame, cv::Mat& rgb_edges,
                         cv::Mat& evt_edge_img) {
        double t_sec = t0 + frameIdx / reader.fps();
        if (!reader.getFrame(t_sec, rgb_frame) || rgb_frame.empty()) return false;

        // Canny on the RGB frame
        cv::Mat gray;
        cv::cvtColor(rgb_frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(3, 3), 1.0);
        cv::Canny(gray, rgb_edges, 40, 120);

        // Accumulate events around this timestamp
        EventBatch events;
        eventReader.readBatch(std::max(0.0, t_sec - evt_window),
                              evt_window * 2, events);
        evt_edge_img = buildEventEdgeImage(events, evt_w, evt_h);

        std::cout << "  Frame " << frameIdx << "  t=" << std::fixed
                  << std::setprecision(3) << t_sec << "s  "
                  << events.size() << " events\n";
        return true;
    };

    cv::Mat rgb_frame, rgb_edges, evt_edge_img;
    if (!loadFrame(rgb_frame, rgb_edges, evt_edge_img)) {
        std::cerr << "Failed to load frame at t=" << args.timeSec << "\n";
        return 1;
    }

    double cur_sx = g_sliders.sx(), cur_sy = g_sliders.sy();
    double cur_ox = g_sliders.ox(), cur_oy = g_sliders.oy();

    // -- Main loop --
    bool running = true;
    while (running) {
        if (g_slider_changed) {
            cur_sx = g_sliders.sx();
            cur_sy = g_sliders.sy();
            cur_ox = g_sliders.ox();
            cur_oy = g_sliders.oy();
            g_slider_changed = false;
        }

        double score;
        cv::Mat composite = buildComposite(rgb_frame, rgb_edges, evt_edge_img,
                                           cur_sx, cur_sy, cur_ox, cur_oy, score);

        char buf[64];
        snprintf(buf, sizeof(buf), "Frame %d/%d", frameIdx, reader.numFrames());
        cv::putText(composite, buf, {composite.cols - 200, 30},
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, {180, 180, 180}, 1);

        cv::imshow(win, composite);

        int key = cv::waitKey(30) & 0xFF;
        switch (key) {
        case 'q': case 27:
            running = false;
            break;

        case 'a': case 'A': {
            // Single-frame auto-alignment
            AlignParams result = gridSearch(rgb_edges, evt_edge_img,
                                            init_sx, init_sy, init_ox, init_oy);
            cur_sx = result.sx; cur_sy = result.sy;
            cur_ox = result.ox; cur_oy = result.oy;
            g_sliders.setFrom(cur_sx, cur_sy, cur_ox, cur_oy);
            cv::setTrackbarPos("Scale X", win, g_sliders.sx_slider);
            cv::setTrackbarPos("Scale Y", win, g_sliders.sy_slider);
            cv::setTrackbarPos("Offset X", win, g_sliders.ox_slider);
            cv::setTrackbarPos("Offset Y", win, g_sliders.oy_slider);
            break;
        }

        case 's': case 'S':
            // Print alignment params in a format you can paste into tracker code
            std::cout << "\n============================================\n"
                      << "  Current alignment parameters:\n"
                      << "    sx = " << std::fixed << std::setprecision(6) << cur_sx << "\n"
                      << "    sy = " << cur_sy << "\n"
                      << "    ox = " << std::setprecision(2) << cur_ox << "\n"
                      << "    oy = " << cur_oy << "\n"
                      << "\n  Copy-paste for tracker_main.cpp:\n"
                      << "    double sr_x = " << std::setprecision(6) << cur_sx << ";\n"
                      << "    double sr_y = " << cur_sy << ";\n"
                      << "    double off_x = " << std::setprecision(2) << cur_ox << ";\n"
                      << "    double off_y = " << cur_oy << ";\n"
                      << "============================================\n\n";
            break;

        case 'n': case 'N':
            frameIdx = std::min(frameIdx + 5, reader.numFrames() - 1);
            loadFrame(rgb_frame, rgb_edges, evt_edge_img);
            break;

        case 'p': case 'P':
            frameIdx = std::max(frameIdx - 5, 0);
            loadFrame(rgb_frame, rgb_edges, evt_edge_img);
            break;

        case 'm': case 'M': {
            // Multi-frame alignment: sample 5 frames spread across the
            // recording and maximize the average score. More robust than
            // single-frame since it's not thrown off by one weird frame.
            std::cout << "\n  Multi-frame auto-alignment (5 frames)...\n";
            auto t_start = std::chrono::steady_clock::now();

            int step = reader.numFrames() / 6;
            std::vector<cv::Mat> rgb_edges_vec, evt_edges_vec;
            for (int f = 1; f <= 5; ++f) {
                int fi = std::min(f * step, reader.numFrames() - 1);
                double t_sec = t0 + fi / reader.fps();
                cv::Mat rgb_f, gray_f, rgb_e, evt_e;
                if (!reader.getFrame(t_sec, rgb_f)) continue;
                cv::cvtColor(rgb_f, gray_f, cv::COLOR_BGR2GRAY);
                cv::GaussianBlur(gray_f, gray_f, cv::Size(3, 3), 1.0);
                cv::Canny(gray_f, rgb_e, 40, 120);

                EventBatch evts;
                eventReader.readBatch(std::max(0.0, t_sec - evt_window),
                                      evt_window * 2, evts);
                evt_e = buildEventEdgeImage(evts, evt_w, evt_h);

                rgb_edges_vec.push_back(rgb_e);
                evt_edges_vec.push_back(evt_e);
                std::cout << "  Loaded frame " << fi << "\n";
            }

            // Same 3-stage search but scoring = average across all frames
            AlignParams best{init_sx, init_sy, init_ox, init_oy, 0.0};
            double best_score = 0;

            // Coarse
            for (double dsx = -0.10; dsx <= 0.10; dsx += 0.02) {
                for (double dsy = -0.10; dsy <= 0.10; dsy += 0.02) {
                    for (double dox = -50; dox <= 50; dox += 10) {
                        for (double doy = -50; doy <= 50; doy += 10) {
                            double sx = init_sx + dsx, sy = init_sy + dsy;
                            double ox = init_ox + dox, oy = init_oy + doy;
                            if (sx < 0.3 || sy < 0.3) continue;
                            double avg = 0;
                            for (size_t i = 0; i < rgb_edges_vec.size(); ++i)
                                avg += scoreAlignment(rgb_edges_vec[i], evt_edges_vec[i],
                                                      sx, sy, ox, oy);
                            avg /= rgb_edges_vec.size();
                            if (avg > best_score) {
                                best_score = avg;
                                best = {sx, sy, ox, oy, avg};
                            }
                        }
                    }
                }
            }

            // Fine
            AlignParams fine = best;
            double fine_score = best.score;
            for (double dsx = -0.02; dsx <= 0.02; dsx += 0.005) {
                for (double dsy = -0.02; dsy <= 0.02; dsy += 0.005) {
                    for (double dox = -12; dox <= 12; dox += 2) {
                        for (double doy = -12; doy <= 12; doy += 2) {
                            double sx = best.sx + dsx, sy = best.sy + dsy;
                            double ox = best.ox + dox, oy = best.oy + doy;
                            double avg = 0;
                            for (size_t i = 0; i < rgb_edges_vec.size(); ++i)
                                avg += scoreAlignment(rgb_edges_vec[i], evt_edges_vec[i],
                                                      sx, sy, ox, oy);
                            avg /= rgb_edges_vec.size();
                            if (avg > fine_score) {
                                fine_score = avg;
                                fine = {sx, sy, ox, oy, avg};
                            }
                        }
                    }
                }
            }

            // Sub-pixel
            AlignParams sub = fine;
            double sub_score = fine.score;
            for (double dsx = -0.003; dsx <= 0.003; dsx += 0.001) {
                for (double dsy = -0.003; dsy <= 0.003; dsy += 0.001) {
                    for (double dox = -3; dox <= 3; dox += 1) {
                        for (double doy = -3; doy <= 3; doy += 1) {
                            double sx = fine.sx + dsx, sy = fine.sy + dsy;
                            double ox = fine.ox + dox, oy = fine.oy + doy;
                            double avg = 0;
                            for (size_t i = 0; i < rgb_edges_vec.size(); ++i)
                                avg += scoreAlignment(rgb_edges_vec[i], evt_edges_vec[i],
                                                      sx, sy, ox, oy);
                            avg /= rgb_edges_vec.size();
                            if (avg > sub_score) {
                                sub_score = avg;
                                sub = {sx, sy, ox, oy, avg};
                            }
                        }
                    }
                }
            }

            auto t_end = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(t_end - t_start).count();

            cur_sx = sub.sx; cur_sy = sub.sy;
            cur_ox = sub.ox; cur_oy = sub.oy;
            g_sliders.setFrom(cur_sx, cur_sy, cur_ox, cur_oy);
            cv::setTrackbarPos("Scale X", win, g_sliders.sx_slider);
            cv::setTrackbarPos("Scale Y", win, g_sliders.sy_slider);
            cv::setTrackbarPos("Offset X", win, g_sliders.ox_slider);
            cv::setTrackbarPos("Offset Y", win, g_sliders.oy_slider);

            std::cout << "  Done in " << std::fixed
                      << std::setprecision(1) << elapsed << "s\n"
                      << "  Score=" << std::setprecision(4) << sub.score
                      << "  sx=" << sub.sx << " sy=" << sub.sy
                      << " ox=" << std::setprecision(1) << sub.ox
                      << " oy=" << sub.oy << "\n\n";
            break;
        }

        default:
            break;
        }
    }

    cv::destroyAllWindows();
    std::cout << "\n  Final parameters:\n"
              << "    sx=" << std::fixed << std::setprecision(6) << cur_sx
              << "  sy=" << cur_sy
              << "  ox=" << std::setprecision(2) << cur_ox
              << "  oy=" << cur_oy << "\n\nDone.\n";
    return 0;
}
