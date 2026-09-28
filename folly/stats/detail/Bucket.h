/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <folly/ConstexprMath.h>
#include <folly/lang/CheckedMath.h>

namespace folly {
namespace detail {

/*
 * Helper function to compute the average, given a specified input type and
 * return type.
 */

// If the input is long double, divide using long double to avoid losing
// precision.
//
// If the ReturnType is integral, the result might be clamped to avoid overflow.
template <typename ReturnType>
ReturnType avgHelper(long double sum, uint64_t count) {
  if (count == 0) {
    return ReturnType(0);
  }
  const long double countf = count;
  if constexpr (std::is_integral<ReturnType>::value) {
    return constexpr_clamp_cast<ReturnType>(sum / countf);
  }
  return static_cast<ReturnType>(sum / countf);
}

// In all other cases divide using double precision.
// This should be relatively fast, and accurate enough for most use cases.
//
// If the ReturnType is integral, the result might be clamped to avoid overflow.
template <typename ReturnType, typename ValueType>
typename std::enable_if<
    !std::is_same<typename std::remove_cv<ValueType>::type, long double>::value,
    ReturnType>::type
avgHelper(ValueType sum, uint64_t count) {
  if (count == 0) {
    return ReturnType(0);
  }
  const double sumf = double(sum);
  const double countf = double(count);
  if constexpr (std::is_integral<ReturnType>::value) {
    return constexpr_clamp_cast<ReturnType>(sumf / countf);
  }
  return static_cast<ReturnType>(sumf / countf);
}

/*
 * Computes value * count, clamping to the numeric limits of ValueType
 * instead of overflowing. count may be any uint64_t, including values
 * wider than ValueType.
 *
 * Floating point types have no overflow concern (the product relaxes to
 * +/-inf), so they fall back to a plain multiply.
 */
template <typename ValueType>
ValueType saturatingMultiply(ValueType value, uint64_t count) {
  if constexpr (!std::is_integral<ValueType>::value) {
    return value * static_cast<ValueType>(count);
  } else if (count == 0 || value == ValueType(0)) {
    return ValueType(0);
  } else {
    // A count wider than ValueType guarantees the product is out of range:
    // |value| >= 1 means |value * count| > max. The one exception is a
    // signed -1, whose product is merely negative, but still below min.
    if constexpr (sizeof(ValueType) < sizeof(uint64_t)) {
      if (count > uint64_t(std::numeric_limits<ValueType>::max())) {
        return value < ValueType(0) ? std::numeric_limits<ValueType>::min()
                                    : std::numeric_limits<ValueType>::max();
      }
    } else if (count > uint64_t(std::numeric_limits<ValueType>::max())) {
      // 64-bit (or wider) ValueType: only count == 2^63 is representable,
      // and only for signed types, where the product of -1 * 2^63 is min.
      if constexpr (std::is_signed<ValueType>::value) {
        if (count == uint64_t(1) << 63 && value == ValueType(-1)) {
          return std::numeric_limits<ValueType>::min();
        }
      }
      return value < ValueType(0) ? std::numeric_limits<ValueType>::min()
                                  : std::numeric_limits<ValueType>::max();
    }
    const auto narrowCount = static_cast<ValueType>(count);
    // folly::checked_mul is portable -- it only uses __builtin_mul_overflow
    // behind FOLLY_HAS_BUILTIN and has a generic fallback -- but it accepts
    // unsigned operands only, so multiply magnitudes and re-apply the sign.
    using U = std::make_unsigned_t<ValueType>;
    constexpr U kMax = static_cast<U>(std::numeric_limits<ValueType>::max());
    const U magnitude = value < ValueType(0)
        ? U(0) - static_cast<U>(value)
        : static_cast<U>(value);
    U product{};
    if (!folly::checked_mul(
            &product, magnitude, static_cast<U>(narrowCount))) {
      // The magnitude of the product does not fit in ValueType at all.
      return value < ValueType(0) ? std::numeric_limits<ValueType>::min()
                                  : std::numeric_limits<ValueType>::max();
    }
    if constexpr (std::is_signed<ValueType>::value) {
      if (value >= ValueType(0)) {
        return product > kMax ? std::numeric_limits<ValueType>::max()
                              : static_cast<ValueType>(product);
      }
      // A negative result may reach -min, whose magnitude is kMax + 1.
      constexpr U kMinMag = kMax + U(1);
      if (product >= kMinMag) {
        return std::numeric_limits<ValueType>::min();
      }
      return static_cast<ValueType>(U(0) - product);
    } else {
      return static_cast<ValueType>(product);
    }
  }
}

// Helpers to add bucket counts and values without
// ever causing undefined behavior
//
// For non-integral vlaues everyhing is easy
template <
    typename ValueType,
    typename std::enable_if<!std::is_integral<ValueType>::value, int>::type = 0>
void addHelper(ValueType& a, const ValueType& b) {
  a += b;
}

template <
    typename ValueType,
    typename std::enable_if<!std::is_integral<ValueType>::value, int>::type = 0>
void subtractHelper(ValueType& a, const ValueType& b) {
  a -= b;
}

// For integral values we use folly/ConstexprMath.h to make
// the math safe and predictable
template <
    typename ValueType,
    typename std::enable_if<std::is_integral<ValueType>::value, int>::type = 0>
void addHelper(ValueType& a, const ValueType& b) {
  a = constexpr_add_overflow_clamped(a, b);
}

template <
    typename ValueType,
    typename std::enable_if<std::is_integral<ValueType>::value, int>::type = 0>
void subtractHelper(ValueType& a, const ValueType& b) {
  a = constexpr_sub_overflow_clamped(a, b);
}

/*
 * Applies a +/- (magnitude * nSamples) to the accumulator in a single
 * saturating step, for signed ValueType. Equivalent to performing nSamples
 * sequential clamped additions (or subtractions) of an addend with the given
 * magnitude and sign: because the addend is constant, the running total
 * moves monotonically and saturates at the same limit either way.
 *
 * All arithmetic is done on uint64_t magnitudes, so the exact magnitude of
 * ValueType min is representable and no intermediate step can overflow.
 * Runs in O(1), so even the maximum nSamples cannot cause an unbounded loop.
 *
 * magnitude must be the exact magnitude of the addend (2^63 is allowed for
 * 64-bit types, i.e. the magnitude of the most negative value).
 */
template <typename ValueType>
void applyRepeatedAccum(
    ValueType& a, bool negative, uint64_t magnitude, uint64_t nSamples) {
  static_assert(std::is_signed<ValueType>::value, "signed type required");
  if (magnitude == 0 || nSamples == 0) {
    return;
  }
  constexpr uint64_t kMax = uint64_t(std::numeric_limits<ValueType>::max());
  constexpr uint64_t kMinMag = kMax + 1; // exact magnitude of ValueType min
  if (nSamples > UINT64_MAX / magnitude) {
    // The true product is at least 2^64, which dwarfs any accumulator value.
    a = negative ? std::numeric_limits<ValueType>::min()
                 : std::numeric_limits<ValueType>::max();
    return;
  }
  const uint64_t p = magnitude * nSamples;
  if (!negative) {
    // total = a + p
    if (a >= ValueType(0)) {
      a = p > kMax - uint64_t(a) ? std::numeric_limits<ValueType>::max()
                                 : static_cast<ValueType>(uint64_t(a) + p);
    } else {
      const uint64_t am = uint64_t(0) - uint64_t(a);
      if (p >= am) {
        const uint64_t t = p - am;
        a = t > kMax ? std::numeric_limits<ValueType>::max()
                     : static_cast<ValueType>(t);
      } else {
        const uint64_t t = am - p;
        a = t > kMinMag ? std::numeric_limits<ValueType>::min()
                        : static_cast<ValueType>(uint64_t(0) - t);
      }
    }
  } else {
    // total = a - p
    if (a >= ValueType(0)) {
      const uint64_t av = uint64_t(a);
      if (av >= p) {
        const uint64_t t = av - p;
        a = t > kMax ? std::numeric_limits<ValueType>::max()
                     : static_cast<ValueType>(t);
      } else {
        const uint64_t t = p - av;
        a = t > kMinMag ? std::numeric_limits<ValueType>::min()
                        : static_cast<ValueType>(uint64_t(0) - t);
      }
    } else {
      const uint64_t am = uint64_t(0) - uint64_t(a);
      if (p > kMinMag - am) {
        a = std::numeric_limits<ValueType>::min();
      } else {
        const uint64_t t = am + p; // <= kMinMag here
        a = t == kMinMag ? std::numeric_limits<ValueType>::min()
                         : static_cast<ValueType>(uint64_t(0) - t);
      }
    }
  }
}

/*
 * Repeatedly add the same value nSamples times to accumulator a,
 * clamping at the numeric limits instead of overflowing. O(1).
 */
template <typename ValueType>
void repeatedValueHelper(ValueType& a, ValueType value, uint64_t nSamples) {
  if (value == ValueType(0) || nSamples == 0) {
    return;
  }
  if constexpr (std::is_integral<ValueType>::value) {
    if constexpr (std::is_signed<ValueType>::value) {
      const bool negative = value < ValueType(0);
      // Negate in unsigned space: -ValueType min is not representable.
      const uint64_t magnitude =
          negative ? uint64_t(0) - uint64_t(value) : uint64_t(value);
      applyRepeatedAccum(a, negative, magnitude, nSamples);
    } else {
      const uint64_t magnitude = uint64_t(value);
      if (nSamples > UINT64_MAX / magnitude) {
        a = std::numeric_limits<ValueType>::max();
        return;
      }
      const uint64_t p = magnitude * nSamples;
      a = p > uint64_t(std::numeric_limits<ValueType>::max()) - uint64_t(a)
          ? std::numeric_limits<ValueType>::max()
          : static_cast<ValueType>(uint64_t(a) + p);
    }
  } else {
    // Floating point: sums relax to +/-inf rather than overflowing, so a
    // single clamped add of value * nSamples gives an O(1) repeated add.
    detail::addHelper(a, value * static_cast<ValueType>(nSamples));
  }
}

/*
 * Repeatedly subtract the same value nSamples times from accumulator a,
 * clamping at the numeric limits instead of overflowing. O(1).
 */
template <typename ValueType>
void subtractRepeatedHelper(
    ValueType& a,
    ValueType value,
    uint64_t nSamples) {
  if (value == ValueType(0) || nSamples == 0) {
    return;
  }
  if constexpr (std::is_integral<ValueType>::value) {
    if constexpr (std::is_signed<ValueType>::value) {
      const bool negative = value < ValueType(0);
      const uint64_t magnitude =
          negative ? uint64_t(0) - uint64_t(value) : uint64_t(value);
      // Subtracting value n times == adding (-value) n times.
      applyRepeatedAccum(a, !negative, magnitude, nSamples);
    } else {
      const uint64_t magnitude = uint64_t(value);
      if (nSamples > UINT64_MAX / magnitude) {
        a = ValueType(0);
        return;
      }
      const uint64_t p = magnitude * nSamples;
      a = p >= uint64_t(a) ? ValueType(0)
                           : static_cast<ValueType>(uint64_t(a) - p);
    }
  } else {
    // Floating point: no overflow concern, so a single clamped subtract of
    // value * nSamples gives an O(1) repeated subtract.
    detail::subtractHelper(a, value * static_cast<ValueType>(nSamples));
  }
}

/*
 * Helper function to compute the rate per Interval,
 * given the specified count recorded over the elapsed time period.
 */
template <
    typename ReturnType = double,
    typename Duration = std::chrono::seconds,
    typename Interval = Duration>
  requires std::constructible_from<Duration, Interval>
ReturnType rateHelper(ReturnType count, Duration elapsed) {
  if (elapsed == Duration(0)) {
    return 0;
  }

  // elapsed is non-zero, and we increase elapsed to at least Interval{1}
  // for rate calculation to smooth out rate calculation at the beginning for
  // a timeseries' lifecycle, when elapsed is very short compared to the overall
  // timeseries duration.
  elapsed = std::max(elapsed, Duration(Interval{1}));

  // Use std::chrono::duration_cast to convert between the native
  // duration and the desired interval.  However, convert the rates,
  // rather than just converting the elapsed duration.  Converting the
  // elapsed time first may collapse it down to 0 if the elapsed interval
  // is less than the desired interval, which will incorrectly result in
  // an infinite rate.
  using NativeRate = std::chrono::duration<
      double,
      std::ratio<Duration::period::den, Duration::period::num>>;
  using DesiredRate = std::chrono::duration<
      double,
      std::ratio<Interval::period::den, Interval::period::num>>;

  // We use Rep=double to avoid rouding down too much here.
  NativeRate native((double)count / elapsed.count());
  DesiredRate desired = std::chrono::duration_cast<DesiredRate>(native);
  return static_cast<ReturnType>(desired.count());
}

template <typename T>
struct Bucket {
 public:
  using ValueType = T;

  Bucket() : sum(ValueType()), count(0) {}

  void clear() {
    sum = ValueType();
    count = 0;
  }

  void add(const ValueType& s, uint64_t c) {
    addHelper(sum, s);
    addHelper(count, c);
  }

  Bucket& operator+=(const Bucket& o) {
    add(o.sum, o.count);
    return *this;
  }

  Bucket& operator-=(const Bucket& o) {
    subtractHelper(sum, o.sum);
    subtractHelper(count, o.count);
    return *this;
  }

  template <typename ReturnType>
  ReturnType avg() const {
    return avgHelper<ReturnType>(sum, count);
  }

  ValueType sum;
  uint64_t count;
};
} // namespace detail
} // namespace folly
