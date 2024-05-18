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
#include <algorithm>
#include <iostream>
#include <cassert>

using std::pair;
using std::vector;
using std::string;

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
  {128, 128, 0},   // 18 olive
  {255, 215, 180}, // 19 apricot
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
    gdImageLine(img_, x0, y0, x1, y1, colors_[color]);
  }

  void rect(int x0, int y0, int x1, int y1, int color) {
    gdImageFilledRectangle(img_, x0, y0, x1, y1, colors_[color]);
  }

  void text(int x, int y, const char *s, int color) {
    gdImageString(img_, font_, x, y, (unsigned char *) s, colors_[color]);
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
  const vector<string> list_;
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


class Anal {
public:
  void addPath(const char *path) { paths_.push_back(path); }

  void run() {
    uint64_t first = 0xffffffffffffffff;
    uint64_t last = 0;
    for (Iter it(paths_); it; ++it) {
      const Rec &r = *it;
      first = std::min(first, r.t0);
      last = std::max(last, r.t1);
    }
    uint64_t dur = last - first;

    std::map<uint32_t, size_t> threads;
    std::map<string, size_t> names;
    for (Iter it(paths_); it; ++it) {
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

    size_t thrSiz = threads.size();
    size_t nameSiz = names.size();

    int wid = 3500;
    int hit0 = 500;
    int hit1 = 750;
    int gap = 24;
    int halfgap = gap / 2;
    double yscale = hit0 / (thrSiz + 2.0);
    double xscale = (wid - gap) / (dur + 2.0);
    int halfy = yscale / 2;
    Img im(wid, hit1);

    // per-thread timelines
    for (size_t i = 0; i < thrSiz; ++i) {
      string ns = std::to_string(i);
      int y = yscale * (i + 2) - halfy;
      im.text(halfgap, y, ns.c_str(), 1); // white
    }

    for (Iter it(paths_); it; ++it) {
      const Rec &r = *it;
      int y = yscale * (threads[r.thread] + 2);
      int x0 = gap + xscale * (r.t0 - first);
      int x1 = gap + xscale * (r.t1 - first);
      int color = names[r.name] + 2;
      im.line(x0, y, x1, y, color);
    }
    double ww = wid / (nameSiz + 2);
    for (const auto &[name, idx] : names) {
      im.text(ww * (idx + 1), halfy, name.c_str(), idx + 2);
    }

    // in-flight graph
    vector<pair<uint64_t, int>> events;
    for (Iter it(paths_); it; ++it) {
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
    double lscale = (hit1 - hit0) / (maxlev + 2.0);
    level = 0;
    uint64_t prev = events[0].first;
    for (auto [time, start] : events) {
      int x0 = gap + xscale * (prev - first);
      int x1 = gap + xscale * (time - first);
      int y = lscale * level;
      im.rect(x0, hit1 - 1 - y, x1, hit1 - 1, 21); // gray
      level = (start ? level + 1 : level - 1);
      prev = time;
    }
    int halfl = lscale / 2;
    for (int i = 1; i <= maxlev; ++i) {
      string ns = std::to_string(i);
      int y = lscale * i;
      im.text(halfgap, hit1 - 1 - halfl - y, ns.c_str(), 1); // white
    }

    im.outPng("im.png");
  }

private:
  vector<string> paths_;
};

int main(int argc, char **argv) {
  Anal aa;
  for (int ii = 1; ii < argc; ++ii)
    aa.addPath(argv[ii]);
  aa.run();
  return 0;
}
