#include "capture/screen_capture.h"

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/XShm.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#include <cstdlib>
#include <cstring>
#include <opencv2/imgproc.hpp>

#include "common/logger.h"

namespace vlm {

struct ScreenCapture::Impl {
  Display* display = nullptr;
  Window root = 0;
  XShmSegmentInfo shm_info{};
  XImage* ximage = nullptr;
  bool use_shm = false;

  ~Impl() { Teardown(); }

  void Teardown() {
    if (use_shm && ximage) {
      XShmDetach(display, &shm_info);
      XDestroyImage(ximage);
      shmdt(shm_info.shmaddr);
      shmctl(shm_info.shmid, IPC_RMID, nullptr);
      ximage = nullptr;
    } else if (ximage) {
      XDestroyImage(ximage);
      ximage = nullptr;
    }
    if (display) {
      XCloseDisplay(display);
      display = nullptr;
    }
  }
};

ScreenCapture::ScreenCapture(Config cfg) : impl_(std::make_unique<Impl>()), cfg_(std::move(cfg)) {}
ScreenCapture::~ScreenCapture() = default;

bool ScreenCapture::Open() {
  const char* dpy_name = cfg_.display.empty() ? std::getenv("DISPLAY") : cfg_.display.c_str();
  if (!dpy_name) dpy_name = ":0";

  impl_->display = XOpenDisplay(dpy_name);
  if (!impl_->display) {
    VLM_LOG_ERROR("capture", "XOpenDisplay failed for ", dpy_name);
    return false;
  }
  impl_->root = DefaultRootWindow(impl_->display);

  XWindowAttributes attrs{};
  XGetWindowAttributes(impl_->display, impl_->root, &attrs);
  if (cfg_.width  <= 0) cfg_.width  = attrs.width  - cfg_.x;
  if (cfg_.height <= 0) cfg_.height = attrs.height - cfg_.y;

  impl_->use_shm = XShmQueryExtension(impl_->display);
  if (impl_->use_shm) {
    impl_->ximage = XShmCreateImage(
        impl_->display,
        attrs.visual,
        attrs.depth,
        ZPixmap,
        nullptr,
        &impl_->shm_info,
        cfg_.width,
        cfg_.height);
    if (!impl_->ximage) {
      VLM_LOG_WARN("capture", "XShmCreateImage failed; falling back to XGetImage");
      impl_->use_shm = false;
    } else {
      impl_->shm_info.shmid =
          shmget(IPC_PRIVATE,
                 static_cast<size_t>(impl_->ximage->bytes_per_line) * impl_->ximage->height,
                 IPC_CREAT | 0600);
      impl_->shm_info.shmaddr = impl_->ximage->data =
          static_cast<char*>(shmat(impl_->shm_info.shmid, nullptr, 0));
      impl_->shm_info.readOnly = False;
      if (!XShmAttach(impl_->display, &impl_->shm_info)) {
        VLM_LOG_WARN("capture", "XShmAttach failed; falling back");
        impl_->use_shm = false;
        XDestroyImage(impl_->ximage);
        impl_->ximage = nullptr;
      }
      XSync(impl_->display, False);
    }
  }

  open_.store(true);
  VLM_LOG_INFO("capture",
               "opened display ", dpy_name,
               " ", cfg_.width, "x", cfg_.height,
               " shm=", impl_->use_shm);
  return true;
}

void ScreenCapture::Close() {
  if (!open_.exchange(false)) return;
  impl_->Teardown();
}

bool ScreenCapture::Grab(cv::Mat& out) {
  if (!open_.load()) return false;

  XImage* img = impl_->ximage;
  if (impl_->use_shm) {
    if (!XShmGetImage(impl_->display, impl_->root, img, cfg_.x, cfg_.y, AllPlanes)) {
      VLM_LOG_ERROR("capture", "XShmGetImage failed");
      return false;
    }
  } else {
    img = XGetImage(impl_->display, impl_->root,
                    cfg_.x, cfg_.y, cfg_.width, cfg_.height,
                    AllPlanes, ZPixmap);
    if (!img) return false;
  }

  // X11 returns BGRA on most modern visuals (depth 24 / 32). Wrap as cv::Mat
  // and drop alpha.
  cv::Mat bgra(img->height, img->width, CV_8UC4, img->data, img->bytes_per_line);
  cv::cvtColor(bgra, out, cv::COLOR_BGRA2BGR);

  if (!impl_->use_shm) {
    XDestroyImage(img);
  }
  return true;
}

}  // namespace vlm
