// Tests for the small enum helpers in common/frame.h. The header only
// defines an enum + a ToString helper, so the tests are correspondingly
// short.

#include "common/frame.h"

#include <gtest/gtest.h>

#include <string>

namespace vlm {
namespace {

TEST(CaptureMode, ToStringCoversAllNamedStates) {
  EXPECT_STREQ(ToString(CaptureMode::kIdle),   "idle");
  EXPECT_STREQ(ToString(CaptureMode::kTyping), "typing");
  EXPECT_STREQ(ToString(CaptureMode::kQuery),  "query");
}

TEST(CaptureMode, ToStringForOutOfRangeReturnsUnknown) {
  // Defensive: the switch falls through to "unknown" if the enum gains
  // a new value that isn't handled. We simulate that with a cast.
  const auto bogus = static_cast<CaptureMode>(99);
  EXPECT_STREQ(ToString(bogus), "unknown");
}

TEST(Frame, DefaultsAreZeroed) {
  Frame f;
  EXPECT_TRUE(f.image.empty());
  EXPECT_EQ(f.seq, 0u);
  EXPECT_EQ(f.ts.time_since_epoch().count(), 0);
}

}  // namespace
}  // namespace vlm
