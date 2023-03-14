// tvm target: c -keys=arm_cpu,cpu -device=arm_cpu -march=armv7e-m -mcpu=cortex-m7 -model=stm32f746xx
#define TVM_EXPORTS
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>
#include <stdbool.h>


#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <arm_acle.h>

#include <tvm/runtime/crt/error_codes.h>


#ifndef ARM_CPU_INTRINSICS_EXIST
#define ARM_CPU_INTRINSICS_EXIST
__attribute__((always_inline)) uint32_t __ror(uint32_t op1, uint32_t op2)
{
  op2 %= 32U;
  if (op2 == 0U)
  {
    return op1;
  }
  return (op1 >> op2) | (op1 << (32U - op2));
}

#define __pkhbt(ARG1,ARG2,ARG3) __extension__ ({                            uint32_t __RES, __ARG1 = (ARG1), __ARG2 = (ARG2);   __asm("pkhbt %0, %1, %2, lsl %3" : "=r" (__RES) :  "r" (__ARG1), "r" (__ARG2), "I" (ARG3)  );   __RES;  })

#define __pkhtb(ARG1,ARG2,ARG3) __extension__ ({                            uint32_t __RES, __ARG1 = (ARG1), __ARG2 = (ARG2);   if (ARG3 == 0)     __asm("pkhtb %0, %1, %2" : "=r" (__RES) :  "r" (__ARG1), "r" (__ARG2)  );   else     __asm("pkhtb %0, %1, %2, asr %3" : "=r" (__RES) :  "r" (__ARG1), "r" (__ARG2), "I" (ARG3)  );   __RES;  })
#endif

#ifndef ARM_CPU_MPROFILE_READ_AND_PAD_EXISTS
#define ARM_CPU_MPROFILE_READ_AND_PAD_EXISTS
__attribute__((always_inline)) static inline const int8_t *read_and_pad(const int8_t *source, int32_t *out1, int32_t *out2)
{
    int32_t inA;
    memcpy(&inA, source, 4);
    source += 4;

    int32_t inAbuf1 = __sxtb16(__ror((uint32_t)inA, 8));
    int32_t inAbuf2 = __sxtb16(inA);
    *out2 = (int32_t)(__pkhtb(inAbuf1, inAbuf2, 16));
    *out1 = (int32_t)(__pkhbt(inAbuf2, inAbuf1, 16));

    return source;
}
#endif



#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm_1x1_body_rest_XEIXXLMK(
    int K,
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int k_base = (K / 4) * 4;
  switch ( K % 4 ) {
  case 1:
    for (int i = 0; i < 1; i++) {
      for (int j = 0; j < 1; j++) {
        int8_t *a_ptr = &aa[i * A_stride + k_base];
        int8_t *b_ptr = &bb[j * B_stride + k_base];
        cc[i * C_stride + j] = (int32_t) a_ptr[0] * (int32_t) b_ptr[0];
      }
    }
    break;
  case 2:
    for (int i = 0; i < 1; i++) {
      for (int j = 0; j < 1; j++) {
        int8_t *a_ptr = &aa[i * A_stride + k_base];
        int8_t *b_ptr = &bb[j * B_stride + k_base];
        cc[i * C_stride + j] =   (int32_t) a_ptr[0] * (int32_t) b_ptr[0]
                               + (int32_t) a_ptr[1] * (int32_t) b_ptr[1];
      }
    }
    break;
  case 3:
    for (int i = 0; i < 1; i++) {
      for (int j = 0; j < 1; j++) {
        int8_t *a_ptr = &aa[i * A_stride + k_base];
        int8_t *b_ptr = &bb[j * B_stride + k_base];
        cc[i * C_stride + j] =   (int32_t) a_ptr[0] * (int32_t) b_ptr[0]
                               + (int32_t) a_ptr[1] * (int32_t) b_ptr[1]
                               + (int32_t) a_ptr[2] * (int32_t) b_ptr[2];
      }
    }
    break;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm_1x1x1_body_loop_XEIXXLMK(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 1; l++) {
        sum += (int32_t) aa[i*A_stride + l] * (int32_t) bb[j*B_stride + l];
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm_1x1x1_body_XEIXXLMK(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t bb_pad[1];
  int32_t retcode = 0;

  if ( 1 < 2 && 1 < 2 ) {
    retcode = gemm_1x1x1_body_loop_XEIXXLMK(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 1; i++)
    for (int j = 0; j < 1 / 4; j++)
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*1 + j*4], (int32_t*) &bb_pad[i*1 + j*4 + 2]);

  for (int i = 0; i < 1; i++) {
    int16_t aa_pad_line[1];
    for (int l = 0; l < 1 / 4; l++)
      read_and_pad(&aa[i*A_stride + l*4], (int32_t*) &aa_pad_line[l*4], (int32_t*) &aa_pad_line[l*4 + 2]);

    for (int j = 0; j < 1; j++) {
      int32_t *aa_ptr = (int32_t *) aa_pad_line;
      int32_t *bb_ptr = (int32_t *) &bb_pad[j*1];
      int32_t sum = 0;
      for (int l = 0; l < 2 * (1 / 4); l++) {
        sum = __smlad(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }

  if ( 1 % 4 != 0 )
    gemm_1x1_body_rest_XEIXXLMK(1, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm_1x1_update_rest_XEIXXLMK(
    int K,
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int k_base = (K / 4) * 4;
  switch ( K % 4 ) {
  case 1:
    for (int i = 0; i < 1; i++) {
      for (int j = 0; j < 1; j++) {
        int8_t *a_ptr = &aa[i * A_stride + k_base];
        int8_t *b_ptr = &bb[j * B_stride + k_base];
        cc[i * C_stride + j] += (int32_t) a_ptr[0] * (int32_t) b_ptr[0];
      }
    }
    break;
  case 2:
    for (int i = 0; i < 1; i++) {
      for (int j = 0; j < 1; j++) {
        int8_t *a_ptr = &aa[i * A_stride + k_base];
        int8_t *b_ptr = &bb[j * B_stride + k_base];
        cc[i * C_stride + j] +=   (int32_t) a_ptr[0] * (int32_t) b_ptr[0]
                                + (int32_t) a_ptr[1] * (int32_t) b_ptr[1];
      }
    }
    break;
  case 3:
    for (int i = 0; i < 1; i++) {
      for (int j = 0; j < 1; j++) {
        int8_t *a_ptr = &aa[i * A_stride + k_base];
        int8_t *b_ptr = &bb[j * B_stride + k_base];
        cc[i * C_stride + j] +=   (int32_t) a_ptr[0] * (int32_t) b_ptr[0]
                                + (int32_t) a_ptr[1] * (int32_t) b_ptr[1]
                                + (int32_t) a_ptr[2] * (int32_t) b_ptr[2];
      }
    }
    break;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm_1x1x1_update_loop_XEIXXLMK(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 1; l++) {
        sum += (int32_t) aa[i*A_stride + l] * (int32_t) bb[j*B_stride + l];
      }
      cc[i*C_stride + j] += sum;
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm_1x1x1_update_XEIXXLMK(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t bb_pad[1];
  int32_t retcode = 0;

  if ( 1 < 2 && 1 < 2 ) {
    retcode = gemm_1x1x1_update_loop_XEIXXLMK(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 1; i++)
    for (int j = 0; j < 1 / 4; j++)
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*1 + j*4], (int32_t*) &bb_pad[i*1 + j*4 + 2]);

  for (int i = 0; i < 1; i++) {
    int16_t aa_pad_line[1];
    for (int l = 0; l < 1 / 4; l++)
      read_and_pad(&aa[i*A_stride + l*4], (int32_t*) &aa_pad_line[l*4], (int32_t*) &aa_pad_line[l*4 + 2]);

    for (int j = 0; j < 1; j++) {
      int32_t *aa_ptr = (int32_t *) aa_pad_line;
      int32_t *bb_ptr = (int32_t *) &bb_pad[j*1];
      int32_t sum = 0;
      for (int l = 0; l < 2 * (1 / 4); l++) {
        sum = __smlad(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      cc[i*C_stride + j] += sum;
    }
  }

  if ( 1 % 4 != 0 )
    gemm_1x1_update_rest_XEIXXLMK(1, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm16_1x1_body_rest_XEIXXLMK(
    int K,
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int k_base = (K / 2) * 2;
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
      int16_t *a_ptr = &aa[i * A_stride + k_base];
      int16_t *b_ptr = &bb[j * B_stride + k_base];
      cc[i * C_stride + j] = (int32_t) a_ptr[0] * (int32_t) b_ptr[0];
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm16_1x1x1_body_loop_XEIXXLMK(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 1; l++) {
        sum += (int32_t) aa[i*A_stride + l] * (int32_t) bb[j*B_stride + l];
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm16_1x1x1_body_XEIXXLMK(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int32_t retcode = 0;

  if ( 1 < 2 && 1 < 2 ) {
    retcode = gemm16_1x1x1_body_loop_XEIXXLMK(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  if(((uint32_t)aa & 0x3) != 0 || ((uint32_t)bb & 0x3) != 0){
    retcode = kTvmErrorFunctionCallInvalidArg;
    goto out;
  }

  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
      int32_t aa_vector[1 / 2];
      int32_t bb_vector[1 / 2];
      memcpy(&aa_vector, &aa[i * A_stride], sizeof(aa_vector));
      memcpy(&bb_vector, &bb[j * B_stride], sizeof(bb_vector));

      int32_t sum = 0;
      for (int l = 0; l < 1 / 2; l++) {
        sum = __smlad(aa_vector[l], bb_vector[l], sum);
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }

  if ( 1 % 2 != 0 )
    gemm16_1x1_body_rest_XEIXXLMK(1, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm16_1x1_update_rest_XEIXXLMK(
    int K,
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int k_base = (K / 2) * 2;
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
      int16_t *a_ptr = &aa[i * A_stride + k_base];
      int16_t *b_ptr = &bb[j * B_stride + k_base];
      cc[i * C_stride + j] += (int32_t) a_ptr[0] * (int32_t) b_ptr[0];
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm16_1x1x1_update_loop_XEIXXLMK(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 1; l++) {
        sum += (int32_t) aa[i*A_stride + l] * (int32_t) bb[j*B_stride + l];
      }
      cc[i*C_stride + j] += sum;
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm16_1x1x1_update_XEIXXLMK(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int32_t retcode = 0;

  if ( 1 < 2 && 1 < 2 ) {
    retcode = gemm16_1x1x1_update_loop_XEIXXLMK(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
      int32_t aa_vector[1 / 2];
      int32_t bb_vector[1 / 2];
      memcpy(&aa_vector, &aa[i * A_stride], sizeof(aa_vector));
      memcpy(&bb_vector, &bb[j * B_stride], sizeof(bb_vector));

      int32_t sum = 0;
      for (int l = 0; l < 1 / 2; l++) {
        sum = __smlad(aa_vector[l], bb_vector[l], sum);
      }
      cc[i*C_stride + j] += sum;
    }
  }

  if ( 1 % 2 != 0 )
    gemm16_1x1_update_rest_XEIXXLMK(1, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm_1x1x1_reset_XEIXXLMK(int32_t *cc, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
      cc[i*C_stride + j] = 0;
    }
  }
  return 0;
}



#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <arm_acle.h>

#include <tvm/runtime/crt/error_codes.h>


#ifndef ARM_CPU_INTRINSICS_EXIST
#define ARM_CPU_INTRINSICS_EXIST
__attribute__((always_inline)) uint32_t __ror(uint32_t op1, uint32_t op2)
{
  op2 %= 32U;
  if (op2 == 0U)
  {
    return op1;
  }
  return (op1 >> op2) | (op1 << (32U - op2));
}

#define __pkhbt(ARG1,ARG2,ARG3) __extension__ ({                            uint32_t __RES, __ARG1 = (ARG1), __ARG2 = (ARG2);   __asm("pkhbt %0, %1, %2, lsl %3" : "=r" (__RES) :  "r" (__ARG1), "r" (__ARG2), "I" (ARG3)  );   __RES;  })

#define __pkhtb(ARG1,ARG2,ARG3) __extension__ ({                            uint32_t __RES, __ARG1 = (ARG1), __ARG2 = (ARG2);   if (ARG3 == 0)     __asm("pkhtb %0, %1, %2" : "=r" (__RES) :  "r" (__ARG1), "r" (__ARG2)  );   else     __asm("pkhtb %0, %1, %2, asr %3" : "=r" (__RES) :  "r" (__ARG1), "r" (__ARG2), "I" (ARG3)  );   __RES;  })
#endif

#ifndef ARM_CPU_MPROFILE_READ_AND_PAD_EXISTS
#define ARM_CPU_MPROFILE_READ_AND_PAD_EXISTS
__attribute__((always_inline)) static inline const int8_t *read_and_pad(const int8_t *source, int32_t *out1, int32_t *out2)
{
    int32_t inA;
    memcpy(&inA, source, 4);
    source += 4;

    int32_t inAbuf1 = __sxtb16(__ror((uint32_t)inA, 8));
    int32_t inAbuf2 = __sxtb16(inA);
    *out2 = (int32_t)(__pkhtb(inAbuf1, inAbuf2, 16));
    *out1 = (int32_t)(__pkhbt(inAbuf2, inAbuf1, 16));

    return source;
}
#endif



#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm_1x1_body_rest_GUIEFLLL(
    int K,
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int k_base = (K / 4) * 4;
  switch ( K % 4 ) {
  case 1:
    for (int i = 0; i < 1; i++) {
      for (int j = 0; j < 1; j++) {
        int8_t *a_ptr = &aa[i * A_stride + k_base];
        int8_t *b_ptr = &bb[j * B_stride + k_base];
        cc[i * C_stride + j] = (int32_t) a_ptr[0] * (int32_t) b_ptr[0];
      }
    }
    break;
  case 2:
    for (int i = 0; i < 1; i++) {
      for (int j = 0; j < 1; j++) {
        int8_t *a_ptr = &aa[i * A_stride + k_base];
        int8_t *b_ptr = &bb[j * B_stride + k_base];
        cc[i * C_stride + j] =   (int32_t) a_ptr[0] * (int32_t) b_ptr[0]
                               + (int32_t) a_ptr[1] * (int32_t) b_ptr[1];
      }
    }
    break;
  case 3:
    for (int i = 0; i < 1; i++) {
      for (int j = 0; j < 1; j++) {
        int8_t *a_ptr = &aa[i * A_stride + k_base];
        int8_t *b_ptr = &bb[j * B_stride + k_base];
        cc[i * C_stride + j] =   (int32_t) a_ptr[0] * (int32_t) b_ptr[0]
                               + (int32_t) a_ptr[1] * (int32_t) b_ptr[1]
                               + (int32_t) a_ptr[2] * (int32_t) b_ptr[2];
      }
    }
    break;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm_1x1x1_body_loop_GUIEFLLL(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 1; l++) {
        sum += (int32_t) aa[i*A_stride + l] * (int32_t) bb[j*B_stride + l];
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm_1x1x1_body_GUIEFLLL(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t bb_pad[1];
  int32_t retcode = 0;

  if ( 1 < 2 && 1 < 2 ) {
    retcode = gemm_1x1x1_body_loop_GUIEFLLL(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 1; i++)
    for (int j = 0; j < 1 / 4; j++)
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*1 + j*4], (int32_t*) &bb_pad[i*1 + j*4 + 2]);

  for (int i = 0; i < 1; i++) {
    int16_t aa_pad_line[1];
    for (int l = 0; l < 1 / 4; l++)
      read_and_pad(&aa[i*A_stride + l*4], (int32_t*) &aa_pad_line[l*4], (int32_t*) &aa_pad_line[l*4 + 2]);

    for (int j = 0; j < 1; j++) {
      int32_t *aa_ptr = (int32_t *) aa_pad_line;
      int32_t *bb_ptr = (int32_t *) &bb_pad[j*1];
      int32_t sum = 0;
      for (int l = 0; l < 2 * (1 / 4); l++) {
        sum = __smlad(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }

  if ( 1 % 4 != 0 )
    gemm_1x1_body_rest_GUIEFLLL(1, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm_1x1_update_rest_GUIEFLLL(
    int K,
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int k_base = (K / 4) * 4;
  switch ( K % 4 ) {
  case 1:
    for (int i = 0; i < 1; i++) {
      for (int j = 0; j < 1; j++) {
        int8_t *a_ptr = &aa[i * A_stride + k_base];
        int8_t *b_ptr = &bb[j * B_stride + k_base];
        cc[i * C_stride + j] += (int32_t) a_ptr[0] * (int32_t) b_ptr[0];
      }
    }
    break;
  case 2:
    for (int i = 0; i < 1; i++) {
      for (int j = 0; j < 1; j++) {
        int8_t *a_ptr = &aa[i * A_stride + k_base];
        int8_t *b_ptr = &bb[j * B_stride + k_base];
        cc[i * C_stride + j] +=   (int32_t) a_ptr[0] * (int32_t) b_ptr[0]
                                + (int32_t) a_ptr[1] * (int32_t) b_ptr[1];
      }
    }
    break;
  case 3:
    for (int i = 0; i < 1; i++) {
      for (int j = 0; j < 1; j++) {
        int8_t *a_ptr = &aa[i * A_stride + k_base];
        int8_t *b_ptr = &bb[j * B_stride + k_base];
        cc[i * C_stride + j] +=   (int32_t) a_ptr[0] * (int32_t) b_ptr[0]
                                + (int32_t) a_ptr[1] * (int32_t) b_ptr[1]
                                + (int32_t) a_ptr[2] * (int32_t) b_ptr[2];
      }
    }
    break;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm_1x1x1_update_loop_GUIEFLLL(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 1; l++) {
        sum += (int32_t) aa[i*A_stride + l] * (int32_t) bb[j*B_stride + l];
      }
      cc[i*C_stride + j] += sum;
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm_1x1x1_update_GUIEFLLL(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t bb_pad[1];
  int32_t retcode = 0;

  if ( 1 < 2 && 1 < 2 ) {
    retcode = gemm_1x1x1_update_loop_GUIEFLLL(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 1; i++)
    for (int j = 0; j < 1 / 4; j++)
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*1 + j*4], (int32_t*) &bb_pad[i*1 + j*4 + 2]);

  for (int i = 0; i < 1; i++) {
    int16_t aa_pad_line[1];
    for (int l = 0; l < 1 / 4; l++)
      read_and_pad(&aa[i*A_stride + l*4], (int32_t*) &aa_pad_line[l*4], (int32_t*) &aa_pad_line[l*4 + 2]);

    for (int j = 0; j < 1; j++) {
      int32_t *aa_ptr = (int32_t *) aa_pad_line;
      int32_t *bb_ptr = (int32_t *) &bb_pad[j*1];
      int32_t sum = 0;
      for (int l = 0; l < 2 * (1 / 4); l++) {
        sum = __smlad(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      cc[i*C_stride + j] += sum;
    }
  }

  if ( 1 % 4 != 0 )
    gemm_1x1_update_rest_GUIEFLLL(1, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm16_1x1_body_rest_GUIEFLLL(
    int K,
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int k_base = (K / 2) * 2;
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
      int16_t *a_ptr = &aa[i * A_stride + k_base];
      int16_t *b_ptr = &bb[j * B_stride + k_base];
      cc[i * C_stride + j] = (int32_t) a_ptr[0] * (int32_t) b_ptr[0];
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm16_1x1x1_body_loop_GUIEFLLL(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 1; l++) {
        sum += (int32_t) aa[i*A_stride + l] * (int32_t) bb[j*B_stride + l];
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm16_1x1x1_body_GUIEFLLL(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int32_t retcode = 0;

  if ( 1 < 2 && 1 < 2 ) {
    retcode = gemm16_1x1x1_body_loop_GUIEFLLL(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  if(((uint32_t)aa & 0x3) != 0 || ((uint32_t)bb & 0x3) != 0){
    retcode = kTvmErrorFunctionCallInvalidArg;
    goto out;
  }

  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
      int32_t aa_vector[1 / 2];
      int32_t bb_vector[1 / 2];
      memcpy(&aa_vector, &aa[i * A_stride], sizeof(aa_vector));
      memcpy(&bb_vector, &bb[j * B_stride], sizeof(bb_vector));

      int32_t sum = 0;
      for (int l = 0; l < 1 / 2; l++) {
        sum = __smlad(aa_vector[l], bb_vector[l], sum);
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }

  if ( 1 % 2 != 0 )
    gemm16_1x1_body_rest_GUIEFLLL(1, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm16_1x1_update_rest_GUIEFLLL(
    int K,
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int k_base = (K / 2) * 2;
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
      int16_t *a_ptr = &aa[i * A_stride + k_base];
      int16_t *b_ptr = &bb[j * B_stride + k_base];
      cc[i * C_stride + j] += (int32_t) a_ptr[0] * (int32_t) b_ptr[0];
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm16_1x1x1_update_loop_GUIEFLLL(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 1; l++) {
        sum += (int32_t) aa[i*A_stride + l] * (int32_t) bb[j*B_stride + l];
      }
      cc[i*C_stride + j] += sum;
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm16_1x1x1_update_GUIEFLLL(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int32_t retcode = 0;

  if ( 1 < 2 && 1 < 2 ) {
    retcode = gemm16_1x1x1_update_loop_GUIEFLLL(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
      int32_t aa_vector[1 / 2];
      int32_t bb_vector[1 / 2];
      memcpy(&aa_vector, &aa[i * A_stride], sizeof(aa_vector));
      memcpy(&bb_vector, &bb[j * B_stride], sizeof(bb_vector));

      int32_t sum = 0;
      for (int l = 0; l < 1 / 2; l++) {
        sum = __smlad(aa_vector[l], bb_vector[l], sum);
      }
      cc[i*C_stride + j] += sum;
    }
  }

  if ( 1 % 2 != 0 )
    gemm16_1x1_update_rest_GUIEFLLL(1, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm_1x1x1_reset_GUIEFLLL(int32_t *cc, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
      cc[i*C_stride + j] = 0;
    }
  }
  return 0;
}



#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <arm_acle.h>

#include <tvm/runtime/crt/error_codes.h>


#ifndef ARM_CPU_INTRINSICS_EXIST
#define ARM_CPU_INTRINSICS_EXIST
__attribute__((always_inline)) uint32_t __ror(uint32_t op1, uint32_t op2)
{
  op2 %= 32U;
  if (op2 == 0U)
  {
    return op1;
  }
  return (op1 >> op2) | (op1 << (32U - op2));
}

#define __pkhbt(ARG1,ARG2,ARG3) __extension__ ({                            uint32_t __RES, __ARG1 = (ARG1), __ARG2 = (ARG2);   __asm("pkhbt %0, %1, %2, lsl %3" : "=r" (__RES) :  "r" (__ARG1), "r" (__ARG2), "I" (ARG3)  );   __RES;  })

#define __pkhtb(ARG1,ARG2,ARG3) __extension__ ({                            uint32_t __RES, __ARG1 = (ARG1), __ARG2 = (ARG2);   if (ARG3 == 0)     __asm("pkhtb %0, %1, %2" : "=r" (__RES) :  "r" (__ARG1), "r" (__ARG2)  );   else     __asm("pkhtb %0, %1, %2, asr %3" : "=r" (__RES) :  "r" (__ARG1), "r" (__ARG2), "I" (ARG3)  );   __RES;  })
#endif

#ifndef ARM_CPU_MPROFILE_READ_AND_PAD_EXISTS
#define ARM_CPU_MPROFILE_READ_AND_PAD_EXISTS
__attribute__((always_inline)) static inline const int8_t *read_and_pad(const int8_t *source, int32_t *out1, int32_t *out2)
{
    int32_t inA;
    memcpy(&inA, source, 4);
    source += 4;

    int32_t inAbuf1 = __sxtb16(__ror((uint32_t)inA, 8));
    int32_t inAbuf2 = __sxtb16(inA);
    *out2 = (int32_t)(__pkhtb(inAbuf1, inAbuf2, 16));
    *out1 = (int32_t)(__pkhbt(inAbuf2, inAbuf1, 16));

    return source;
}
#endif



#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm_1x1_body_rest_CNNKGLIC(
    int K,
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int k_base = (K / 4) * 4;
  switch ( K % 4 ) {
  case 1:
    for (int i = 0; i < 1; i++) {
      for (int j = 0; j < 1; j++) {
        int8_t *a_ptr = &aa[i * A_stride + k_base];
        int8_t *b_ptr = &bb[j * B_stride + k_base];
        cc[i * C_stride + j] = (int32_t) a_ptr[0] * (int32_t) b_ptr[0];
      }
    }
    break;
  case 2:
    for (int i = 0; i < 1; i++) {
      for (int j = 0; j < 1; j++) {
        int8_t *a_ptr = &aa[i * A_stride + k_base];
        int8_t *b_ptr = &bb[j * B_stride + k_base];
        cc[i * C_stride + j] =   (int32_t) a_ptr[0] * (int32_t) b_ptr[0]
                               + (int32_t) a_ptr[1] * (int32_t) b_ptr[1];
      }
    }
    break;
  case 3:
    for (int i = 0; i < 1; i++) {
      for (int j = 0; j < 1; j++) {
        int8_t *a_ptr = &aa[i * A_stride + k_base];
        int8_t *b_ptr = &bb[j * B_stride + k_base];
        cc[i * C_stride + j] =   (int32_t) a_ptr[0] * (int32_t) b_ptr[0]
                               + (int32_t) a_ptr[1] * (int32_t) b_ptr[1]
                               + (int32_t) a_ptr[2] * (int32_t) b_ptr[2];
      }
    }
    break;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm_1x1x1_body_loop_CNNKGLIC(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 1; l++) {
        sum += (int32_t) aa[i*A_stride + l] * (int32_t) bb[j*B_stride + l];
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm_1x1x1_body_CNNKGLIC(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t bb_pad[1];
  int32_t retcode = 0;

  if ( 1 < 2 && 1 < 2 ) {
    retcode = gemm_1x1x1_body_loop_CNNKGLIC(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 1; i++)
    for (int j = 0; j < 1 / 4; j++)
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*1 + j*4], (int32_t*) &bb_pad[i*1 + j*4 + 2]);

  for (int i = 0; i < 1; i++) {
    int16_t aa_pad_line[1];
    for (int l = 0; l < 1 / 4; l++)
      read_and_pad(&aa[i*A_stride + l*4], (int32_t*) &aa_pad_line[l*4], (int32_t*) &aa_pad_line[l*4 + 2]);

    for (int j = 0; j < 1; j++) {
      int32_t *aa_ptr = (int32_t *) aa_pad_line;
      int32_t *bb_ptr = (int32_t *) &bb_pad[j*1];
      int32_t sum = 0;
      for (int l = 0; l < 2 * (1 / 4); l++) {
        sum = __smlad(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }

  if ( 1 % 4 != 0 )
    gemm_1x1_body_rest_CNNKGLIC(1, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm_1x1_update_rest_CNNKGLIC(
    int K,
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int k_base = (K / 4) * 4;
  switch ( K % 4 ) {
  case 1:
    for (int i = 0; i < 1; i++) {
      for (int j = 0; j < 1; j++) {
        int8_t *a_ptr = &aa[i * A_stride + k_base];
        int8_t *b_ptr = &bb[j * B_stride + k_base];
        cc[i * C_stride + j] += (int32_t) a_ptr[0] * (int32_t) b_ptr[0];
      }
    }
    break;
  case 2:
    for (int i = 0; i < 1; i++) {
      for (int j = 0; j < 1; j++) {
        int8_t *a_ptr = &aa[i * A_stride + k_base];
        int8_t *b_ptr = &bb[j * B_stride + k_base];
        cc[i * C_stride + j] +=   (int32_t) a_ptr[0] * (int32_t) b_ptr[0]
                                + (int32_t) a_ptr[1] * (int32_t) b_ptr[1];
      }
    }
    break;
  case 3:
    for (int i = 0; i < 1; i++) {
      for (int j = 0; j < 1; j++) {
        int8_t *a_ptr = &aa[i * A_stride + k_base];
        int8_t *b_ptr = &bb[j * B_stride + k_base];
        cc[i * C_stride + j] +=   (int32_t) a_ptr[0] * (int32_t) b_ptr[0]
                                + (int32_t) a_ptr[1] * (int32_t) b_ptr[1]
                                + (int32_t) a_ptr[2] * (int32_t) b_ptr[2];
      }
    }
    break;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm_1x1x1_update_loop_CNNKGLIC(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 1; l++) {
        sum += (int32_t) aa[i*A_stride + l] * (int32_t) bb[j*B_stride + l];
      }
      cc[i*C_stride + j] += sum;
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm_1x1x1_update_CNNKGLIC(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t bb_pad[1];
  int32_t retcode = 0;

  if ( 1 < 2 && 1 < 2 ) {
    retcode = gemm_1x1x1_update_loop_CNNKGLIC(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 1; i++)
    for (int j = 0; j < 1 / 4; j++)
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*1 + j*4], (int32_t*) &bb_pad[i*1 + j*4 + 2]);

  for (int i = 0; i < 1; i++) {
    int16_t aa_pad_line[1];
    for (int l = 0; l < 1 / 4; l++)
      read_and_pad(&aa[i*A_stride + l*4], (int32_t*) &aa_pad_line[l*4], (int32_t*) &aa_pad_line[l*4 + 2]);

    for (int j = 0; j < 1; j++) {
      int32_t *aa_ptr = (int32_t *) aa_pad_line;
      int32_t *bb_ptr = (int32_t *) &bb_pad[j*1];
      int32_t sum = 0;
      for (int l = 0; l < 2 * (1 / 4); l++) {
        sum = __smlad(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      cc[i*C_stride + j] += sum;
    }
  }

  if ( 1 % 4 != 0 )
    gemm_1x1_update_rest_CNNKGLIC(1, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm16_1x1_body_rest_CNNKGLIC(
    int K,
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int k_base = (K / 2) * 2;
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
      int16_t *a_ptr = &aa[i * A_stride + k_base];
      int16_t *b_ptr = &bb[j * B_stride + k_base];
      cc[i * C_stride + j] = (int32_t) a_ptr[0] * (int32_t) b_ptr[0];
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm16_1x1x1_body_loop_CNNKGLIC(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 1; l++) {
        sum += (int32_t) aa[i*A_stride + l] * (int32_t) bb[j*B_stride + l];
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm16_1x1x1_body_CNNKGLIC(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int32_t retcode = 0;

  if ( 1 < 2 && 1 < 2 ) {
    retcode = gemm16_1x1x1_body_loop_CNNKGLIC(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  if(((uint32_t)aa & 0x3) != 0 || ((uint32_t)bb & 0x3) != 0){
    retcode = kTvmErrorFunctionCallInvalidArg;
    goto out;
  }

  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
      int32_t aa_vector[1 / 2];
      int32_t bb_vector[1 / 2];
      memcpy(&aa_vector, &aa[i * A_stride], sizeof(aa_vector));
      memcpy(&bb_vector, &bb[j * B_stride], sizeof(bb_vector));

      int32_t sum = 0;
      for (int l = 0; l < 1 / 2; l++) {
        sum = __smlad(aa_vector[l], bb_vector[l], sum);
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }

  if ( 1 % 2 != 0 )
    gemm16_1x1_body_rest_CNNKGLIC(1, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm16_1x1_update_rest_CNNKGLIC(
    int K,
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int k_base = (K / 2) * 2;
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
      int16_t *a_ptr = &aa[i * A_stride + k_base];
      int16_t *b_ptr = &bb[j * B_stride + k_base];
      cc[i * C_stride + j] += (int32_t) a_ptr[0] * (int32_t) b_ptr[0];
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm16_1x1x1_update_loop_CNNKGLIC(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 1; l++) {
        sum += (int32_t) aa[i*A_stride + l] * (int32_t) bb[j*B_stride + l];
      }
      cc[i*C_stride + j] += sum;
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm16_1x1x1_update_CNNKGLIC(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int32_t retcode = 0;

  if ( 1 < 2 && 1 < 2 ) {
    retcode = gemm16_1x1x1_update_loop_CNNKGLIC(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
      int32_t aa_vector[1 / 2];
      int32_t bb_vector[1 / 2];
      memcpy(&aa_vector, &aa[i * A_stride], sizeof(aa_vector));
      memcpy(&bb_vector, &bb[j * B_stride], sizeof(bb_vector));

      int32_t sum = 0;
      for (int l = 0; l < 1 / 2; l++) {
        sum = __smlad(aa_vector[l], bb_vector[l], sum);
      }
      cc[i*C_stride + j] += sum;
    }
  }

  if ( 1 % 2 != 0 )
    gemm16_1x1_update_rest_CNNKGLIC(1, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t gemm_1x1x1_reset_CNNKGLIC(int32_t *cc, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
      cc[i*C_stride + j] = 0;
    }
  }
  return 0;
}



#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <arm_acle.h>

#include <tvm/runtime/crt/error_codes.h>


#ifndef ARM_CPU_INTRINSICS_EXIST
#define ARM_CPU_INTRINSICS_EXIST
__attribute__((always_inline)) uint32_t __ror(uint32_t op1, uint32_t op2)
{
  op2 %= 32U;
  if (op2 == 0U)
  {
    return op1;
  }
  return (op1 >> op2) | (op1 << (32U - op2));
}

#define __pkhbt(ARG1,ARG2,ARG3) __extension__ ({                            uint32_t __RES, __ARG1 = (ARG1), __ARG2 = (ARG2);   __asm("pkhbt %0, %1, %2, lsl %3" : "=r" (__RES) :  "r" (__ARG1), "r" (__ARG2), "I" (ARG3)  );   __RES;  })

#define __pkhtb(ARG1,ARG2,ARG3) __extension__ ({                            uint32_t __RES, __ARG1 = (ARG1), __ARG2 = (ARG2);   if (ARG3 == 0)     __asm("pkhtb %0, %1, %2" : "=r" (__RES) :  "r" (__ARG1), "r" (__ARG2)  );   else     __asm("pkhtb %0, %1, %2, asr %3" : "=r" (__RES) :  "r" (__ARG1), "r" (__ARG2), "I" (ARG3)  );   __RES;  })
#endif



#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t max8_reset_UERLZRAY(
    int8_t *res,
    int N) {
  memset(res, (int8_t)-128, N * sizeof(*res));
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t max8_loop_UERLZRAY(
    int8_t *arg,
    int8_t *res,
    int N) {
  for ( int i = 0; i < N; ++ i )
    if ( arg[i] > res[i] )
      res[i] = arg[i];
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t max8_UERLZRAY(
    int8_t *arg,
    int8_t *res,
    int N) {
  int32_t *parg32, *pres32;
  int una_arg = (int32_t)arg & 0x3, una_res = (int32_t)res & 0x3;
  int32_t retcode = 0;

  if ( N < 4 || ((una_arg || una_res) && una_arg != una_res) ) {
    retcode = max8_loop_UERLZRAY(arg, res, N);
    goto out;
  }
  if ( una_arg ) {
    int n = (4 - una_arg);
    if ( n > N || (N - n) < 4 )
      n = N;
    retcode = max8_loop_UERLZRAY(arg, res, n);
    N -= n;
    if ( N == 0 )
      goto out;
    arg += n; res += n;
  }

  parg32 = (int32_t *)arg;
  pres32 = (int32_t *)res;

  for ( int i = 0; i < N / 4; ++ i ) {
    int32_t arg32 = *parg32 ++;
    int32_t res32 = *pres32;
    __ssub8(arg32, res32);
    res32 = __sel(arg32, res32);
    *pres32 ++ = res32;
  }

  if ( N & 0x3 ) {
    retcode = max8_loop_UERLZRAY((int8_t *)parg32, (int8_t *)pres32, N & 0x3);
    goto out;
  }

out:
  return retcode;
}



#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <arm_acle.h>

#include <tvm/runtime/crt/error_codes.h>


#ifndef ARM_CPU_INTRINSICS_EXIST
#define ARM_CPU_INTRINSICS_EXIST
__attribute__((always_inline)) uint32_t __ror(uint32_t op1, uint32_t op2)
{
  op2 %= 32U;
  if (op2 == 0U)
  {
    return op1;
  }
  return (op1 >> op2) | (op1 << (32U - op2));
}

#define __pkhbt(ARG1,ARG2,ARG3) __extension__ ({                            uint32_t __RES, __ARG1 = (ARG1), __ARG2 = (ARG2);   __asm("pkhbt %0, %1, %2, lsl %3" : "=r" (__RES) :  "r" (__ARG1), "r" (__ARG2), "I" (ARG3)  );   __RES;  })

#define __pkhtb(ARG1,ARG2,ARG3) __extension__ ({                            uint32_t __RES, __ARG1 = (ARG1), __ARG2 = (ARG2);   if (ARG3 == 0)     __asm("pkhtb %0, %1, %2" : "=r" (__RES) :  "r" (__ARG1), "r" (__ARG2)  );   else     __asm("pkhtb %0, %1, %2, asr %3" : "=r" (__RES) :  "r" (__ARG1), "r" (__ARG2), "I" (ARG3)  );   __RES;  })
#endif



#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t max8_reset_OUYSNXKO(
    int8_t *res,
    int N) {
  memset(res, (int8_t)-128, N * sizeof(*res));
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t max8_loop_OUYSNXKO(
    int8_t *arg,
    int8_t *res,
    int N) {
  for ( int i = 0; i < N; ++ i )
    if ( arg[i] > res[i] )
      res[i] = arg[i];
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__attribute__((always_inline)) static inline int32_t max8_OUYSNXKO(
    int8_t *arg,
    int8_t *res,
    int N) {
  int32_t *parg32, *pres32;
  int una_arg = (int32_t)arg & 0x3, una_res = (int32_t)res & 0x3;
  int32_t retcode = 0;

  if ( N < 4 || ((una_arg || una_res) && una_arg != una_res) ) {
    retcode = max8_loop_OUYSNXKO(arg, res, N);
    goto out;
  }
  if ( una_arg ) {
    int n = (4 - una_arg);
    if ( n > N || (N - n) < 4 )
      n = N;
    retcode = max8_loop_OUYSNXKO(arg, res, n);
    N -= n;
    if ( N == 0 )
      goto out;
    arg += n; res += n;
  }

  parg32 = (int32_t *)arg;
  pres32 = (int32_t *)res;

  for ( int i = 0; i < N / 4; ++ i ) {
    int32_t arg32 = *parg32 ++;
    int32_t res32 = *pres32;
    __ssub8(arg32, res32);
    res32 = __sel(arg32, res32);
    *pres32 ++ = res32;
  }

  if ( N & 0x3 ) {
    retcode = max8_loop_OUYSNXKO((int8_t *)parg32, (int8_t *)pres32, N & 0x3);
    goto out;
  }

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_divide_round_add_clip_cast_reshape_cast(float* p0, int16_t* T_cast, uint8_t* global_const_workspace_2_var, uint8_t* global_workspace_3_var) {
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 784; ++ax0_ax1_fused_ax2_fused) {
    float v_ = roundf((p0[ax0_ax1_fused_ax2_fused] * 2.550092e+02f)) + -1.280000e+02f;
    float v__1 = (v_) < (1.270000e+02f) ? (v_) : (1.270000e+02f);
    T_cast[ax0_ax1_fused_ax2_fused] = ((int16_t)((int8_t)((v__1) > (-1.280000e+02f) ? (v__1) : (-1.280000e+02f))));
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_conv2d_subtract_add_fixed_point_multiply_per_axis_add_clip_cast(int16_t* p0, int8_t* T_cast, uint8_t* global_const_workspace_4_var, uint8_t* global_workspace_5_var) {
  void* fused_nn_conv2d_subtract_add_constant_2_let = (&(global_const_workspace_4_var[48832]));
  void* fused_nn_conv2d_subtract_add_constant_1_let = (&(global_const_workspace_4_var[48864]));
  void* fused_nn_conv2d_subtract_add_constant_let = (&(global_const_workspace_4_var[48896]));
  void* fused_nn_conv2d_constant_let = (&(global_const_workspace_4_var[48928]));
  void* fused_nn_conv2d_subtract_constant_let = (&(global_const_workspace_4_var[48800]));
  void* fused_constant_let = (&(global_const_workspace_4_var[48080]));
  for (int32_t ax0_ax1_outer_fused = 0; ax0_ax1_outer_fused < 24; ++ax0_ax1_outer_fused) {
    void* PadInput_let = (&(global_workspace_5_var[5216]));
    void* conv_let = (&(global_workspace_5_var[5024]));
    for (int32_t ax2_outer = 0; ax2_outer < 3; ++ax2_outer) {
      for (int32_t i1 = 0; i1 < 5; ++i1) {
        for (int32_t i2 = 0; i2 < 12; ++i2) {
          ((int16_t*)PadInput_let)[((i1 * 12) + i2)] = p0[((((i1 * 28) + (ax0_ax1_outer_fused * 28)) + (ax2_outer * 8)) + i2)];
        }
      }
      for (int32_t oco = 0; oco < 6; ++oco) {
        for (int32_t owi_init = 0; owi_init < 8; ++owi_init) {
          ((int32_t*)conv_let)[((oco * 8) + owi_init)] = 0;
        }
        for (int32_t kh = 0; kh < 5; ++kh) {
          for (int32_t kw = 0; kw < 5; ++kw) {
            for (int32_t owi = 0; owi < 8; ++owi) {
              int32_t cse_var_1 = ((oco * 8) + owi);
              ((int32_t*)conv_let)[cse_var_1] = (((int32_t*)conv_let)[cse_var_1] + (((int32_t)((int16_t*)PadInput_let)[(((kh * 12) + owi) + kw)]) * ((int32_t)((int16_t*)fused_constant_let)[(((kh * 30) + (kw * 6)) + oco)])));
            }
          }
        }
      }
      for (int32_t ax3_outer = 0; ax3_outer < 6; ++ax3_outer) {
        for (int32_t ax2_inner = 0; ax2_inner < 8; ++ax2_inner) {
          int32_t v_ = ((int32_t)(((((int64_t)((((int32_t*)conv_let)[((ax3_outer * 8) + ax2_inner)] + ((int32_t*)fused_nn_conv2d_subtract_constant_let)[ax3_outer]) - ((int32_t*)fused_nn_conv2d_constant_let)[ax3_outer])) * ((int64_t)((int32_t*)fused_nn_conv2d_subtract_add_constant_let)[ax3_outer])) + ((int64_t)1 << ((int64_t)((((int32_t*)fused_nn_conv2d_subtract_add_constant_2_let)[ax3_outer] + 31) - 1)))) >> ((int64_t)(((int32_t*)fused_nn_conv2d_subtract_add_constant_2_let)[ax3_outer] + 31)))) - 128;
          int32_t v__1 = (v_) < (127) ? (v_) : (127);
          T_cast[((((ax0_ax1_outer_fused * 144) + (ax2_outer * 48)) + (ax2_inner * 6)) + ax3_outer)] = ((int8_t)((v__1) > (-128) ? (v__1) : (-128)));
        }
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_conv2d_subtract_add_fixed_point_multiply_per_axis_add_clip_cast_1(int16_t* p0, int8_t* T_cast, uint8_t* global_const_workspace_8_var, uint8_t* global_workspace_9_var) {
  void* fused_nn_conv2d_subtract_add_constant_5_let = (&(global_const_workspace_8_var[48448]));
  void* fused_nn_conv2d_subtract_add_constant_4_let = (&(global_const_workspace_8_var[48512]));
  void* fused_nn_conv2d_subtract_add_constant_3_let = (&(global_const_workspace_8_var[48576]));
  void* fused_nn_conv2d_constant_1_let = (&(global_const_workspace_8_var[48640]));
  void* fused_nn_conv2d_subtract_constant_1_let = (&(global_const_workspace_8_var[48384]));
  void* fused_constant_1_let = (&(global_const_workspace_8_var[40800]));
  for (int32_t ax0_ax1_outer_fused = 0; ax0_ax1_outer_fused < 8; ++ax0_ax1_outer_fused) {
    void* PadInput_let = (&(global_workspace_9_var[2752]));
    void* conv_let = (&(global_workspace_9_var[3472]));
    for (int32_t i1 = 0; i1 < 5; ++i1) {
      for (int32_t i2 = 0; i2 < 12; ++i2) {
        for (int32_t i3 = 0; i3 < 6; ++i3) {
          int32_t cse_var_2 = (i1 * 72);
          int32_t cse_var_1 = (i2 * 6);
          ((int16_t*)PadInput_let)[((cse_var_2 + cse_var_1) + i3)] = p0[(((cse_var_2 + (ax0_ax1_outer_fused * 72)) + cse_var_1) + i3)];
        }
      }
    }
    for (int32_t oco = 0; oco < 2; ++oco) {
      for (int32_t owi_init = 0; owi_init < 8; ++owi_init) {
        for (int32_t oci_init = 0; oci_init < 8; ++oci_init) {
          ((int32_t*)conv_let)[(((oco * 64) + (owi_init * 8)) + oci_init)] = 0;
        }
      }
      for (int32_t kh = 0; kh < 5; ++kh) {
        for (int32_t kw = 0; kw < 5; ++kw) {
          for (int32_t ic = 0; ic < 6; ++ic) {
            for (int32_t owi = 0; owi < 8; ++owi) {
              for (int32_t oci = 0; oci < 8; ++oci) {
                int32_t cse_var_3 = (((oco * 64) + (owi * 8)) + oci);
                ((int32_t*)conv_let)[cse_var_3] = (((int32_t*)conv_let)[cse_var_3] + (((int32_t)((int16_t*)PadInput_let)[((((kh * 72) + (owi * 6)) + (kw * 6)) + ic)]) * ((int32_t)((int16_t*)fused_constant_1_let)[(((((kh * 480) + (kw * 96)) + (ic * 16)) + (oco * 8)) + oci)])));
              }
            }
          }
        }
      }
    }
    for (int32_t ax3_outer = 0; ax3_outer < 2; ++ax3_outer) {
      for (int32_t ax2_inner = 0; ax2_inner < 8; ++ax2_inner) {
        for (int32_t ax3_inner = 0; ax3_inner < 8; ++ax3_inner) {
          int32_t cse_var_5 = (ax3_outer * 8);
          int32_t cse_var_4 = (cse_var_5 + ax3_inner);
          int32_t v_ = ((int32_t)(((((int64_t)((((int32_t*)conv_let)[(((ax3_outer * 64) + (ax2_inner * 8)) + ax3_inner)] + ((int32_t*)fused_nn_conv2d_subtract_constant_1_let)[cse_var_4]) - ((int32_t*)fused_nn_conv2d_constant_1_let)[cse_var_4])) * ((int64_t)((int32_t*)fused_nn_conv2d_subtract_add_constant_3_let)[cse_var_4])) + ((int64_t)1 << ((int64_t)((((int32_t*)fused_nn_conv2d_subtract_add_constant_5_let)[cse_var_4] + 31) - 1)))) >> ((int64_t)(((int32_t*)fused_nn_conv2d_subtract_add_constant_5_let)[cse_var_4] + 31)))) - 128;
          int32_t v__1 = (v_) < (127) ? (v_) : (127);
          T_cast[((((ax0_ax1_outer_fused * 128) + (ax2_inner * 16)) + cse_var_5) + ax3_inner)] = ((int8_t)((v__1) > (-128) ? (v__1) : (-128)));
        }
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_dense_subtract_add_fixed_point_multiply_add_clip_cast(int8_t* p0, int8_t* T_cast, uint8_t* global_const_workspace_14_var, uint8_t* global_workspace_15_var) {
  void* fused_nn_dense_constant_let = (&(global_const_workspace_14_var[46928]));
  void* fused_nn_dense_subtract_constant_let = (&(global_const_workspace_14_var[46448]));
  void* fused_constant_2_let = (&(global_const_workspace_14_var[0]));
  void* dense_let = (&(global_workspace_15_var[0]));
  for (int32_t y_outer = 0; y_outer < 120; ++y_outer) {
    gemm_1x1x1_reset_XEIXXLMK((&(((int32_t*)dense_let)[y_outer])), 1);
    for (int32_t k_outer = 0; k_outer < 256; ++k_outer) {
      gemm_1x1x1_update_XEIXXLMK((&(p0[k_outer])), (&(((int8_t*)fused_constant_2_let)[((y_outer * 256) + k_outer)])), (&(((int32_t*)dense_let)[y_outer])), 1, 1, 1);
    }
  }
  for (int32_t ax1 = 0; ax1 < 120; ++ax1) {
    int32_t v_ = ((int32_t)(((((0 != 0) ? (((int64_t)((((int32_t*)dense_let)[ax1] + ((int32_t*)fused_nn_dense_subtract_constant_let)[ax1]) - ((int32_t*)fused_nn_dense_constant_let)[ax1])) << ((int64_t)0)) : ((int64_t)((((int32_t*)dense_let)[ax1] + ((int32_t*)fused_nn_dense_subtract_constant_let)[ax1]) - ((int32_t*)fused_nn_dense_constant_let)[ax1]))) * (int64_t)1264522615) + ((int64_t)1 << ((int64_t)((8 + 31) - 1)))) >> ((int64_t)(8 + 31)))) - 128;
    int32_t v__1 = (v_) < (127) ? (v_) : (127);
    T_cast[ax1] = ((int8_t)((v__1) > (-128) ? (v__1) : (-128)));
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_dense_subtract_add_fixed_point_multiply_add_clip_cast_1(int8_t* p0, int8_t* T_cast, uint8_t* global_const_workspace_16_var, uint8_t* global_workspace_17_var) {
  void* fused_nn_dense_constant_1_let = (&(global_const_workspace_16_var[47744]));
  void* fused_nn_dense_subtract_constant_1_let = (&(global_const_workspace_16_var[47408]));
  void* fused_constant_3_let = (&(global_const_workspace_16_var[30720]));
  void* dense_let = (&(global_workspace_17_var[0]));
  for (int32_t y_outer = 0; y_outer < 84; ++y_outer) {
    gemm_1x1x1_reset_GUIEFLLL((&(((int32_t*)dense_let)[y_outer])), 1);
    for (int32_t k_outer = 0; k_outer < 120; ++k_outer) {
      gemm_1x1x1_update_GUIEFLLL((&(p0[k_outer])), (&(((int8_t*)fused_constant_3_let)[((y_outer * 120) + k_outer)])), (&(((int32_t*)dense_let)[y_outer])), 1, 1, 1);
    }
  }
  for (int32_t ax1 = 0; ax1 < 84; ++ax1) {
    int32_t v_ = ((int32_t)(((((0 != 0) ? (((int64_t)((((int32_t*)dense_let)[ax1] + ((int32_t*)fused_nn_dense_subtract_constant_1_let)[ax1]) - ((int32_t*)fused_nn_dense_constant_1_let)[ax1])) << ((int64_t)0)) : ((int64_t)((((int32_t*)dense_let)[ax1] + ((int32_t*)fused_nn_dense_subtract_constant_1_let)[ax1]) - ((int32_t*)fused_nn_dense_constant_1_let)[ax1]))) * (int64_t)1364818348) + ((int64_t)1 << ((int64_t)((8 + 31) - 1)))) >> ((int64_t)(8 + 31)))) - 128;
    int32_t v__1 = (v_) < (127) ? (v_) : (127);
    T_cast[ax1] = ((int8_t)((v__1) > (-128) ? (v__1) : (-128)));
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_dense_subtract_add_fixed_point_multiply_add_clip_subtract_cast_multiply(int8_t* p0, float* T_multiply, uint8_t* global_const_workspace_18_var, uint8_t* global_workspace_19_var) {
  void* fused_nn_dense_constant_2_let = (&(global_const_workspace_18_var[48752]));
  void* fused_nn_dense_subtract_constant_2_let = (&(global_const_workspace_18_var[48704]));
  void* fused_constant_4_let = (&(global_const_workspace_18_var[45600]));
  void* dense_let = (&(global_workspace_19_var[420]));
  for (int32_t y_outer = 0; y_outer < 10; ++y_outer) {
    gemm_1x1x1_reset_CNNKGLIC((&(((int32_t*)dense_let)[y_outer])), 1);
    for (int32_t k_outer = 0; k_outer < 84; ++k_outer) {
      gemm_1x1x1_update_CNNKGLIC((&(p0[k_outer])), (&(((int8_t*)fused_constant_4_let)[((y_outer * 84) + k_outer)])), (&(((int32_t*)dense_let)[y_outer])), 1, 1, 1);
    }
  }
  for (int32_t ax1 = 0; ax1 < 10; ++ax1) {
    int32_t v_ = ((int32_t)(((((0 != 0) ? (((int64_t)((((int32_t*)dense_let)[ax1] + ((int32_t*)fused_nn_dense_subtract_constant_2_let)[ax1]) - ((int32_t*)fused_nn_dense_constant_2_let)[ax1])) << ((int64_t)0)) : ((int64_t)((((int32_t*)dense_let)[ax1] + ((int32_t*)fused_nn_dense_subtract_constant_2_let)[ax1]) - ((int32_t*)fused_nn_dense_constant_2_let)[ax1]))) * (int64_t)1400994510) + ((int64_t)1 << ((int64_t)((8 + 31) - 1)))) >> ((int64_t)(8 + 31)))) - 128;
    int32_t v__1 = (v_) < (127) ? (v_) : (127);
    T_multiply[ax1] = (((float)(((v__1) > (-128) ? (v__1) : (-128)) + 128)) * 1.458327e-02f);
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_max_pool2d(int8_t* p0, int8_t* pool_max, uint8_t* global_const_workspace_10_var, uint8_t* global_workspace_11_var) {
  for (int32_t ax1 = 0; ax1 < 4; ++ax1) {
    for (int32_t ax2 = 0; ax2 < 4; ++ax2) {
      max8_reset_UERLZRAY((&(pool_max[((ax1 * 64) + (ax2 * 16))])), 16);
      for (int32_t rv0 = 0; rv0 < 2; ++rv0) {
        for (int32_t rv1 = 0; rv1 < 2; ++rv1) {
          max8_UERLZRAY((&(p0[((((ax1 * 256) + (rv0 * 128)) + (ax2 * 32)) + (rv1 * 16))])), (&(pool_max[((ax1 * 64) + (ax2 * 16))])), 16);
        }
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_max_pool2d_cast(int8_t* p0, int16_t* T_cast, uint8_t* global_const_workspace_6_var, uint8_t* global_workspace_7_var) {
  void* pool_max_let = (&(global_workspace_7_var[3456]));
  for (int32_t ax1 = 0; ax1 < 12; ++ax1) {
    for (int32_t ax2 = 0; ax2 < 12; ++ax2) {
      max8_reset_OUYSNXKO((&(((int8_t*)pool_max_let)[((ax1 * 72) + (ax2 * 6))])), 6);
      for (int32_t rv0 = 0; rv0 < 2; ++rv0) {
        for (int32_t rv1 = 0; rv1 < 2; ++rv1) {
          max8_OUYSNXKO((&(p0[((((ax1 * 288) + (rv0 * 144)) + (ax2 * 12)) + (rv1 * 6))])), (&(((int8_t*)pool_max_let)[((ax1 * 72) + (ax2 * 6))])), 6);
        }
      }
    }
  }
  for (int32_t ax1_1 = 0; ax1_1 < 12; ++ax1_1) {
    for (int32_t ax2_1 = 0; ax2_1 < 12; ++ax2_1) {
      for (int32_t ax3 = 0; ax3 < 6; ++ax3) {
        int32_t cse_var_1 = (((ax1_1 * 72) + (ax2_1 * 6)) + ax3);
        T_cast[cse_var_1] = ((int16_t)((int8_t*)pool_max_let)[cse_var_1]);
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_transpose_reshape(int8_t* p0, int8_t* T_reshape, uint8_t* global_const_workspace_12_var, uint8_t* global_workspace_13_var) {
  for (int32_t ax1_outer = 0; ax1_outer < 16; ++ax1_outer) {
    for (int32_t ax1_inner = 0; ax1_inner < 16; ++ax1_inner) {
      T_reshape[((ax1_outer * 16) + ax1_inner)] = p0[((ax1_inner * 16) + ax1_outer)];
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default___tvm_main__(float* serving_default_input_0_buffer_var, float* PartitionedCall_0_buffer_var, uint8_t* global_const_workspace_0_var, uint8_t* global_workspace_1_var) {
  void* sid_4_let = (&(global_workspace_1_var[1728]));
  void* sid_3_let = (&(global_workspace_1_var[0]));
  void* sid_5_let = (&(global_workspace_1_var[2752]));
  void* sid_2_let = (&(global_workspace_1_var[0]));
  void* sid_6_let = (&(global_workspace_1_var[480]));
  void* sid_1_let = (&(global_workspace_1_var[3456]));
  void* sid_7_let = (&(global_workspace_1_var[480]));
  void* sid_8_let = (&(global_workspace_1_var[336]));
  if (tvmgen_default_fused_divide_round_add_clip_cast_reshape_cast(serving_default_input_0_buffer_var, sid_1_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_conv2d_subtract_add_fixed_point_multiply_per_axis_add_clip_cast(sid_1_let, sid_2_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_max_pool2d_cast(sid_2_let, sid_3_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_conv2d_subtract_add_fixed_point_multiply_per_axis_add_clip_cast_1(sid_3_let, sid_4_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_max_pool2d(sid_4_let, sid_5_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_transpose_reshape(sid_5_let, sid_6_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_dense_subtract_add_fixed_point_multiply_add_clip_cast(sid_6_let, sid_7_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_dense_subtract_add_fixed_point_multiply_add_clip_cast_1(sid_7_let, sid_8_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_dense_subtract_add_fixed_point_multiply_add_clip_subtract_cast_multiply(sid_8_let, PartitionedCall_0_buffer_var, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  return 0;
}

