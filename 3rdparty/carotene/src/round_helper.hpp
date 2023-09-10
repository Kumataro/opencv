/*
 * By downloading, copying, installing or using the software you agree to this license.
 * If you do not agree to this license, do not download, install,
 * copy or use the software.
 *
 *
 *                           License Agreement
 *                For Open Source Computer Vision Library
 *                        (3-clause BSD License)
 *
 * Copyright (C) 2014-2015, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *   * Neither the names of the copyright holders nor the names of the contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 * This software is provided by the copyright holders and contributors "as is" and
 * any express or implied warranties, including, but not limited to, the implied
 * warranties of merchantability and fitness for a particular purpose are disclaimed.
 * In no event shall copyright holders or contributors be liable for any direct,
 * indirect, incidental, special, exemplary, or consequential damages
 * (including, but not limited to, procurement of substitute goods or services;
 * loss of use, data, or profits; or business interruption) however caused
 * and on any theory of liability, whether in contract, strict liability,
 * or tort (including negligence or otherwise) arising in any way out of
 * the use of this software, even if advised of the possibility of such damage.
 */

#ifndef CAROTENE_SRC_VROUND_HPP
#define CAROTENE_SRC_VROUND_HPP

#include "common.hpp"
#include "vtransform.hpp"
#include <iostream>

#ifdef CAROTENE_NEON

/**
 * This helper header is for rounding from float32xN to uin32xN or int32xN to nearest, ties to even.
 * See https://en.wikipedia.org/wiki/Rounding#Rounding_half_to_even
 */

namespace CAROTENE_NS { namespace internal {

#if ( defined(__aarch64__) || defined(__aarch32__) ) && !defined(OPENCV_FORCE_TO_USE_ARMV7_NEON )
#  define NEON_SUPPORTS_VCNTN_INSTRUCTION
#elseif defined(OPENCV_FORCE_TO_LEGACY_NEON_ROUNDING)
#  define NEON_SUPPORTS_LEGACY_METHOD
#endif

inline uint32x4_t _neon_round_u32_f32(const float32x4_t val)
{
#ifdef NEON_SUPPORTS_VCNTN_INSTRUCTION
    return vcvtnq_u32_f32(val);
#elseif defined( NEON_SUPPORTS_LEGACY_METHOD )
    return vcvtq_u32_f32(val);
#else
    static const float32x4_t f32_v0_5 = vdupq_n_f32(0.5);
    static const uint32x4_t  u32_v1_0 = vdupq_n_u32(1);

    const uint32x4_t round = vcvtq_u32_f32( vaddq_f32(val, f32_v0_5 ) );
    const uint32x4_t isOdd = vandq_u32( round, u32_v1_0 );
    const uint32x4_t isFrac0_5 = vceqq_f32(vsubq_f32(vcvtq_f32_u32(round),val), f32_v0_5 );
    const uint32x4_t ret = vsubq_u32( round, vandq_u32( isOdd, isFrac0_5 ) );

    return ret;
#endif
}

inline uint32x2_t _neon_round_u32_f32(const float32x2_t val)
{
#ifdef NEON_SUPPORTS_VCNTN_INSTRUCTION
    return vcvtn_u32_f32(val);
#elseif defined( NEON_SUPPORTS_LEGACY_METHOD )
    return vcvt_u32_f32(val);
#else
    static const float32x2_t f32_v0_5 = vdup_n_f32(0.5);
    static const uint32x2_t  u32_v1_0 = vdup_n_u32(1);

    const uint32x2_t round = vcvt_u32_f32( vadd_f32(val, f32_v0_5 ) );
    const uint32x2_t isOdd = vand_u32( round, u32_v1_0 );
    const uint32x2_t isFrac0_5 = vceq_f32(vsub_f32(vcvt_f32_u32(round),val), f32_v0_5 );
    const uint32x2_t ret = vsub_u32( round, vand_u32( isOdd, isFrac0_5 ) );

    return ret;
#endif
}

inline int32x4_t _neon_round_s32_f32(const float32x4_t val)
{
#ifdef NEON_SUPPORTS_VCNTN_INSTRUCTION
    return vcvtnq_s32_f32(val);
#elseif defined( NEON_SUPPORTS_LEGACY_METHOD )
    return vcvtq_u32_f32(val);
#else
    static const float32x4_t f32_v0_0  = vdupq_n_f32(0.0);
    static const int32x4_t   s32_v1_0  = vdupq_n_s32(1);
    static const int32x4_t   s32_vM1_0 = vdupq_n_s32(-1);

    const float32x4_t val_abs = vabsq_f32( val );

    const int32x4_t isNegative = vreinterpretq_s32_u32( vcleq_f32( val, f32_v0_0 ) );
    const int32x4_t ret_signs = vorrq_s32(
        vandq_s32( s32_vM1_0, isNegative ),
        vbicq_s32( s32_v1_0,  isNegative )  // it means Positive
    );

    const int32x4_t ret_abs = vreinterpretq_s32_u32( _neon_round_u32_f32( val_abs ) );
    const int32x4_t ret = vmulq_s32( ret_abs, ret_signs );

    return ret;
#endif
}

inline int32x2_t _neon_round_s32_f32(const float32x2_t val)
{
#ifdef NEON_SUPPORTS_VCNTN_INSTRUCTION
    return vcvtn_s32_f32(val);
#elseif defined( NEON_SUPPORTS_LEGACY_METHOD )
    return vcvt_u32_f32(val);
#else
    static const float32x2_t f32_v0_0  = vdup_n_f32(0.0);
    static const int32x2_t   s32_v1_0  = vdup_n_s32(1);
    static const int32x2_t   s32_vM1_0 = vdup_n_s32(-1);

    const float32x2_t val_abs = vabs_f32( val );

    const int32x2_t isNegative = vreinterpret_s32_u32( vcle_f32( val, f32_v0_0 ) );
    const int32x2_t ret_signs = vorr_s32(
        vand_s32( s32_vM1_0, isNegative ),
        vbic_s32( s32_v1_0,  isNegative )  // it means Positive
    );

    const int32x2_t ret_abs = vreinterpret_s32_u32( _neon_round_u32_f32( val_abs ) );
    const int32x2_t ret = vmul_s32( ret_abs, ret_signs );

    return ret;
#endif
}

} }

#endif // CAROTENE_NEON

#endif
