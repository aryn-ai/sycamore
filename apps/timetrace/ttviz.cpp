// ttviz.cpp - TimeTrace visualizer
//
// requires libgd (aka: gd)

#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <gd.h>
#include <gdfontmb.h>

#include <map>
#include <array>
#include <vector>
#include <string>
#include <string_view>
#include <algorithm>
#include <cassert>

using std::pair;
using std::vector;
using std::string;
using std::string_view;


struct Rgb {
  int r;
  int g;
  int b;
};

const std::array<Rgb, 22> gColors = {{ // from Sasha Trubetskoy
  {0,   0,   0},   //  0 black
  {255, 255, 255}, //  1 white
  {230, 25,  75},  //  2 red
  {60,  180, 75},  //  3 green
  {255, 225, 25},  //  4 yellow
  {0,   130, 200}, //  5 blue
  {245, 130, 48},  //  6 orange
  {145, 30,  180}, //  7 purple
  {70,  240, 240}, //  8 cyan
  {240, 50,  230}, //  9 magenta
  {210, 245, 60},  // 10 lime
  {250, 190, 212}, // 11 pink
  {0,   128, 128}, // 12 teal
  {220, 190, 255}, // 13 lavender
  {170, 110, 40},  // 14 brown
  {255, 250, 200}, // 15 beige
  {128, 0,   0},   // 16 maroon
  {170, 255, 195}, // 17 mint
  {255, 215, 180}, // 18 apricot
  {128, 128, 0},   // 19 olive
  {0,   0,   128}, // 20 navy
  {128, 128, 128}  // 21 gray
  }};


class Img {
public:
  Img(int width, int height) {
    img_ = gdImageCreate(width, height);
    for (const Rgb &c : gColors)
      colors_.push_back(gdImageColorAllocate(img_, c.r, c.g, c.b));
    font_ = gdFontGetMediumBold();
  }

  void line(int x0, int y0, int x1, int y1, int color) {
    gdImageLine(img_, x0, y0, x1, y1, getColor(color));
  }

  void rect(int x0, int y0, int x1, int y1, int color) {
    gdImageFilledRectangle(img_, x0, y0, x1, y1, getColor(color));
  }

  void text(int x, int y, const char *s, int color) {
    gdImageString(img_, font_, x, y, (unsigned char *) s, getColor(color));
  }

  void outPng(const char *path) {
    FILE *fout = fopen(path, "wb");
    gdImagePng(img_, fout);
    fclose(fout);
  }

private:
  gdImagePtr img_;
  gdFontPtr  font_;
  vector<int> colors_;

  int getColor(int c) {
    size_t n = colors_.size();
    if ((size_t) c < n)
      return colors_[c];
    int i = (c % (n - 2)) + 2;  // loop through non black/white colors
    return colors_[i];
  }
};


struct Rec {
  uint8_t  version;
  uint8_t  pad0;
  uint16_t pad1;
  uint32_t thread;
  uint64_t t0;
  uint64_t t1;
  uint64_t utime;
  uint64_t stime;
  uint64_t rss;
  char     name[48];
};


class Iter {
public:
  Iter(const vector<string> &l) : list_(l), idx_(0), fd_(-1) { advance(); }
  ~Iter() { if (fd_ >= 0) close(fd_); }
  Rec& operator*() { return rec_; }
  operator bool() const { return (fd_ >= 0); }
  Iter &operator++() { advance(); return *this; }

private:
  const vector<string> &list_;
  size_t idx_;
  int fd_;
  Rec rec_;

  void advance() {
    for (;;) {
      if (fd_ >= 0) {
        ssize_t nb = read(fd_, &rec_, sizeof(Rec));
        if (nb == sizeof(Rec)) {
          assert(rec_.version == 0);
          return;
        }
        close(fd_);
        fd_ = -1;
      }
      if (idx_ >= list_.size())
        return;
      fd_ = open(list_[idx_].c_str(), O_RDONLY);
      ++idx_;
    }
  }
};


class Viz {
public:
  Viz() : width_(1600), yinc_(18), outPath_("viz.png") {}

  void addPath(const char *path) { inPaths_.push_back(path); }

  void setSize(int width, int yinc) {
    width_ = width;
    yinc_ = yinc;
  }

  void setOutPath(const char *path) { outPath_ = path; }

  void run() {
    // find beginning and end of time
    uint64_t first = 0xffffffffffffffff;
    uint64_t last = 0;
    for (Iter it(inPaths_); it; ++it) {
      const Rec &r = *it;
      first = std::min(first, r.t0);
      last = std::max(last, r.t1);
    }
    uint64_t dur = last - first;

    // collect threads and names
    std::map<uint32_t, size_t> threads;
    std::map<string, size_t> names;
    for (Iter it(inPaths_); it; ++it) {
      const Rec &r = *it;
      if (!threads.contains(r.thread)) {
        size_t n = threads.size();
        threads[r.thread] = n;
      }
      if (!names.contains(r.name)) {
        size_t n = names.size();
        names[r.name] = n;
      }
    }
    size_t thrCnt = threads.size();
    size_t nameCnt = names.size();

    // calculate some dimensions
    int heightTop = yinc_ * thrCnt;
    int heightBot = (yinc_ / 2) * nameCnt;
    if (heightTop < 128)
      heightTop = 128;
    if (heightBot < 64)
      heightBot = 64;
    int heightAll = heightTop + heightBot;
    int xgap = 12;
    double yscale = heightTop / (thrCnt + 2.0);
    double xscale = (width_ - xgap) / (dur + 2.0);
    Img im(width_, heightAll);

    // per-thread timelines
    for (size_t i = 0; i < thrCnt; ++i) {
      string ns = std::to_string(i);
      int y = yscale * (i + 2) - (yscale / 2);
      im.text(xgap / 2, y, ns.c_str(), 1); // white
    }

    for (Iter it(inPaths_); it; ++it) {
      const Rec &r = *it;
      int y = yscale * (threads[r.thread] + 2);
      int x0 = xgap + xscale * (r.t0 - first);
      int x1 = xgap + xscale * (r.t1 - first);
      int idx = names[r.name];
      y += idx - (nameCnt / 2); // avoid overlapping lines
      im.line(x0, y, x1, y, idx + 2);
    }
    double ww = width_ / (nameCnt + 1);
    for (const auto &[name, idx] : names) {
      im.text((ww * idx) + (ww / 2), yscale / 2, name.c_str(), idx + 2);
    }

    // in-flight graph
    vector<pair<uint64_t, int>> events;
    for (Iter it(inPaths_); it; ++it) {
      const Rec &r = *it;
      events.emplace_back(r.t0, 1);
      events.emplace_back(r.t1, 0);
    }
    std::sort(events.begin(), events.end());
    int level = 0;
    int maxlev = 0;
    for (auto [time, start] : events) {
      level = (start ? level + 1 : level - 1);
      maxlev = std::max(maxlev, level);
    }
    double lscale = heightBot / (maxlev + 2.0);
    level = 0;
    uint64_t prev = events[0].first;
    for (auto [time, start] : events) {
      int x0 = xgap + xscale * (prev - first);
      int x1 = xgap + xscale * (time - first);
      int y = lscale * level;
      im.rect(x0, heightAll - 1 - y, x1, heightAll - 1, 21); // 21 = gray
      level = (start ? level + 1 : level - 1);
      prev = time;
    }
    for (int i = 1; i <= maxlev; ++i) {
      string ns = std::to_string(i);
      int y = heightAll - 1 - (lscale / 2) - (lscale * i);
      im.text(xgap / 2, y, ns.c_str(), 1); // 1 = white
    }

    // rss graph
    vector<pair<uint64_t, uint64_t>> samples;
    for (Iter it(inPaths_); it; ++it) {
      const Rec &r = *it;
      samples.emplace_back(r.t1, r.rss);
    }
    std::sort(samples.begin(), samples.end());
    uint64_t maxrss = 0;
    for (auto [time, rss] : samples)
      maxrss = std::max(maxrss, rss);
    double rscale = heightBot / (maxrss + 2.0);
    uint64_t prevtime = first;
    uint64_t prevrss = 0;
    for (auto [time, rss] : samples) {
      int x0 = xgap + xscale * (prevtime - first);
      int x1 = xgap + xscale * (time - first);
      int y0 = heightAll - 1 - rscale * prevrss;
      int y1 = heightAll - 1 - rscale * rss;
      im.line(x0, y0, x1, y1, 1); // 1 = white
      prevtime = time;
      prevrss = rss;
    }

    im.outPng(outPath_.c_str());
  }

private:
  int width_;
  int yinc_;
  string outPath_;
  vector<string> inPaths_;
};


int main(int argc, char **argv) {
  Viz viz;
  for (int ii = 1; ii < argc; ++ii) {
    if (argv[ii][0] != '-')
      viz.addPath(argv[ii]);
    else {
      string_view arg = argv[ii];
      if (arg == "-s")
        viz.setSize(1024, 16);
      else if (arg == "-m")
        viz.setSize(1600, 18);
      else if (arg == "-l")
        viz.setSize(2400, 22);
      else if (arg == "-xl")
        viz.setSize(3600, 24);
      else if ((arg == "-o") && ((ii + 1) < argc))
        viz.setOutPath(argv[ii + 1]);
    }
  }
  viz.run();
  return 0;
}
