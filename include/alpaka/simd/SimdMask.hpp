#pragma once

#include "alpaka/core/Common.hpp"

#include <array>
#include <cstddef>

namespace alpaka::simd
{

    //! SIMD mask type for conditional operations
    template<typename T, typename TAcc>
    class SimdMask
    {
    public:
        using value_type = bool;
        static constexpr std::size_t width = simd_width<T, TAcc>::value;

        ALPAKA_FN_HOST_ACC SimdMask() = default;

        //! Constructor from scalar bool
        ALPAKA_FN_HOST_ACC explicit SimdMask(bool value)
        {
            data_.fill(value);
        }

        //! Element access
        ALPAKA_FN_ACC bool operator[](std::size_t idx) const
        {
            return data_[idx];
        }

        ALPAKA_FN_ACC bool& operator[](std::size_t idx)
        {
            return data_[idx];
        }

        //! Logical operations
        ALPAKA_FN_ACC SimdMask operator&(SimdMask const& rhs) const
        {
            SimdMask result;
            for(std::size_t i = 0; i < width; ++i)
            {
                result.data_[i] = data_[i] && rhs.data_[i];
            }
            return result;
        }

        ALPAKA_FN_ACC SimdMask operator|(SimdMask const& rhs) const
        {
            SimdMask result;
            for(std::size_t i = 0; i < width; ++i)
            {
                result.data_[i] = data_[i] || rhs.data_[i];
            }
            return result;
        }

        ALPAKA_FN_ACC SimdMask operator^(SimdMask const& rhs) const
        {
            SimdMask result;
            for(std::size_t i = 0; i < width; ++i)
            {
                result.data_[i] = data_[i] != rhs.data_[i];
            }
            return result;
        }

        ALPAKA_FN_ACC SimdMask operator~() const
        {
            SimdMask result;
            for(std::size_t i = 0; i < width; ++i)
            {
                result.data_[i] = !data_[i];
            }
            return result;
        }

        //! Compound assignment operators
        ALPAKA_FN_ACC SimdMask& operator&=(SimdMask const& rhs)
        {
            *this = *this & rhs;
            return *this;
        }

        ALPAKA_FN_ACC SimdMask& operator|=(SimdMask const& rhs)
        {
            *this = *this | rhs;
            return *this;
        }

        ALPAKA_FN_ACC SimdMask& operator^=(SimdMask const& rhs)
        {
            *this = *this ^ rhs;
            return *this;
        }

        //! Query operations
        ALPAKA_FN_ACC bool all() const
        {
            for(std::size_t i = 0; i < width; ++i)
            {
                if(!data_[i])
                    return false;
            }
            return true;
        }

        ALPAKA_FN_ACC bool any() const
        {
            for(std::size_t i = 0; i < width; ++i)
            {
                if(data_[i])
                    return true;
            }
            return false;
        }

        ALPAKA_FN_ACC bool none() const
        {
            return !any();
        }

        //! Get number of true elements
        ALPAKA_FN_ACC std::size_t count() const
        {
            std::size_t result = 0;
            for(std::size_t i = 0; i < width; ++i)
            {
                if(data_[i])
                    ++result;
            }
            return result;
        }

    private:
        std::array<bool, width> data_{};
    };

} // namespace alpaka::simd
