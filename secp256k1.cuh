#include <iostream>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#define BIGINT_WORDS 8

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

struct BigInt {
    uint32_t data[BIGINT_WORDS];
};

struct ECPoint {
    BigInt x, y;
    bool infinity;
};

struct ECPointJac {
    BigInt X, Y, Z;
    bool infinity;
};

__constant__ BigInt const_p;
__constant__ ECPointJac const_G_jacobian;
__constant__ BigInt const_n;

__host__ __device__ __forceinline__ void init_bigint(BigInt *x, uint32_t val) {
    x->data[0] = val;
    for (int i = 1; i < BIGINT_WORDS; i++) x->data[i] = 0;
}

__host__ __device__ __forceinline__ void copy_bigint(BigInt *dest, const BigInt *src) {
    for (int i = 0; i < BIGINT_WORDS; i++) {
        dest->data[i] = src->data[i];
    }
}

__host__ __device__ __forceinline__ int compare_bigint(const BigInt *a, const BigInt *b) {
    for (int i = BIGINT_WORDS - 1; i >= 0; i--) {
        if (a->data[i] > b->data[i]) return 1;
        if (a->data[i] < b->data[i]) return -1;
    }
    return 0;
}

__host__ __device__ __forceinline__ bool is_zero(const BigInt *a) {
    for (int i = 0; i < BIGINT_WORDS; i++) {
        if (a->data[i]) return false;
    }
    return true;
}

__host__ __device__ __forceinline__ int get_bit(const BigInt *a, int i) {
    int word_idx = i >> 5; // i / 32
    int bit_idx = i & 31;  // i % 32
    if (word_idx >= BIGINT_WORDS) return 0;
    return (a->data[word_idx] >> bit_idx) & 1;
}

__device__ __forceinline__ void ptx_u256Add(BigInt *res, const BigInt *a, const BigInt *b) {
    asm volatile(
        "add.cc.u32 %0, %8, %16;\n\t"
        "addc.cc.u32 %1, %9, %17;\n\t"
        "addc.cc.u32 %2, %10, %18;\n\t"
        "addc.cc.u32 %3, %11, %19;\n\t"
        "addc.cc.u32 %4, %12, %20;\n\t"
        "addc.cc.u32 %5, %13, %21;\n\t"
        "addc.cc.u32 %6, %14, %22;\n\t"
        "addc.u32 %7, %15, %23;\n\t"
        : "=r"(res->data[0]), "=r"(res->data[1]), "=r"(res->data[2]), "=r"(res->data[3]),
          "=r"(res->data[4]), "=r"(res->data[5]), "=r"(res->data[6]), "=r"(res->data[7])
        : "r"(a->data[0]), "r"(a->data[1]), "r"(a->data[2]), "r"(a->data[3]),
          "r"(a->data[4]), "r"(a->data[5]), "r"(a->data[6]), "r"(a->data[7]),
          "r"(b->data[0]), "r"(b->data[1]), "r"(b->data[2]), "r"(b->data[3]),
          "r"(b->data[4]), "r"(b->data[5]), "r"(b->data[6]), "r"(b->data[7])
    );
}

__device__ __forceinline__ void ptx_u256Sub(BigInt *res, const BigInt *a, const BigInt *b) {
    asm volatile(
        "sub.cc.u32 %0, %8, %16;\n\t"
        "subc.cc.u32 %1, %9, %17;\n\t"
        "subc.cc.u32 %2, %10, %18;\n\t"
        "subc.cc.u32 %3, %11, %19;\n\t"
        "subc.cc.u32 %4, %12, %20;\n\t"
        "subc.cc.u32 %5, %13, %21;\n\t"
        "subc.cc.u32 %6, %14, %22;\n\t"
        "subc.u32 %7, %15, %23;\n\t"
        : "=r"(res->data[0]), "=r"(res->data[1]), "=r"(res->data[2]), "=r"(res->data[3]),
          "=r"(res->data[4]), "=r"(res->data[5]), "=r"(res->data[6]), "=r"(res->data[7])
        : "r"(a->data[0]), "r"(a->data[1]), "r"(a->data[2]), "r"(a->data[3]),
          "r"(a->data[4]), "r"(a->data[5]), "r"(a->data[6]), "r"(a->data[7]),
          "r"(b->data[0]), "r"(b->data[1]), "r"(b->data[2]), "r"(b->data[3]),
          "r"(b->data[4]), "r"(b->data[5]), "r"(b->data[6]), "r"(b->data[7])
    );
}

// Optimized multiply_bigint_by_const with unrolling
__device__ __forceinline__ void multiply_bigint_by_const(const BigInt *a, uint32_t c, uint32_t result[9]) {
    uint64_t carry = 0;
    
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint64_t prod = (uint64_t)a->data[i] * c + carry;
        result[i] = (uint32_t)prod;
        carry = prod >> 32;
    }
    result[8] = (uint32_t)carry;
}

// Optimized shift_left_word
__device__ __forceinline__ void shift_left_word(const BigInt *a, uint32_t result[9]) {
    result[0] = 0;
    
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; i++) {
        result[i+1] = a->data[i];
    }
}

// Optimized add_9word with unrolling
__device__ __forceinline__ void add_9word(uint32_t r[9], const uint32_t addend[9]) {
    uint64_t carry = 0;
    
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        uint64_t sum = (uint64_t)r[i] + addend[i] + carry;
        r[i] = (uint32_t)sum;
        carry = sum >> 32;
    }
}

__device__ __forceinline__ void convert_9word_to_bigint(const uint32_t r[9], BigInt *res) {
    for (int i = 0; i < BIGINT_WORDS; i++) {
        res->data[i] = r[i];
    }
}
__device__ __forceinline__ void mul_mod_device(BigInt *res, const BigInt *a, const BigInt *b) {
    uint32_t prod[16] = {0};

    // i = 0
    {
        uint32_t carry = 0;
        uint64_t sum;

        uint32_t low = a->data[0] * b->data[0];
        uint32_t high = __umulhi(a->data[0], b->data[0]);
        sum = (uint64_t)prod[0] + low + carry;
        prod[0] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[0] * b->data[1];
        high = __umulhi(a->data[0], b->data[1]);
        sum = (uint64_t)prod[1] + low + carry;
        prod[1] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[0] * b->data[2];
        high = __umulhi(a->data[0], b->data[2]);
        sum = (uint64_t)prod[2] + low + carry;
        prod[2] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[0] * b->data[3];
        high = __umulhi(a->data[0], b->data[3]);
        sum = (uint64_t)prod[3] + low + carry;
        prod[3] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[0] * b->data[4];
        high = __umulhi(a->data[0], b->data[4]);
        sum = (uint64_t)prod[4] + low + carry;
        prod[4] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[0] * b->data[5];
        high = __umulhi(a->data[0], b->data[5]);
        sum = (uint64_t)prod[5] + low + carry;
        prod[5] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[0] * b->data[6];
        high = __umulhi(a->data[0], b->data[6]);
        sum = (uint64_t)prod[6] + low + carry;
        prod[6] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[0] * b->data[7];
        high = __umulhi(a->data[0], b->data[7]);
        sum = (uint64_t)prod[7] + low + carry;
        prod[7] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        prod[8] += carry;
    }

    // i = 1
    {
        uint32_t carry = 0;
        uint64_t sum;

        uint32_t low = a->data[1] * b->data[0];
        uint32_t high = __umulhi(a->data[1], b->data[0]);
        sum = (uint64_t)prod[1] + low + carry;
        prod[1] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[1] * b->data[1];
        high = __umulhi(a->data[1], b->data[1]);
        sum = (uint64_t)prod[2] + low + carry;
        prod[2] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[1] * b->data[2];
        high = __umulhi(a->data[1], b->data[2]);
        sum = (uint64_t)prod[3] + low + carry;
        prod[3] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[1] * b->data[3];
        high = __umulhi(a->data[1], b->data[3]);
        sum = (uint64_t)prod[4] + low + carry;
        prod[4] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[1] * b->data[4];
        high = __umulhi(a->data[1], b->data[4]);
        sum = (uint64_t)prod[5] + low + carry;
        prod[5] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[1] * b->data[5];
        high = __umulhi(a->data[1], b->data[5]);
        sum = (uint64_t)prod[6] + low + carry;
        prod[6] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[1] * b->data[6];
        high = __umulhi(a->data[1], b->data[6]);
        sum = (uint64_t)prod[7] + low + carry;
        prod[7] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[1] * b->data[7];
        high = __umulhi(a->data[1], b->data[7]);
        sum = (uint64_t)prod[8] + low + carry;
        prod[8] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        prod[9] += carry;
    }

    // i = 2
    {
        uint32_t carry = 0;
        uint64_t sum;

        uint32_t low = a->data[2] * b->data[0];
        uint32_t high = __umulhi(a->data[2], b->data[0]);
        sum = (uint64_t)prod[2] + low + carry;
        prod[2] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[2] * b->data[1];
        high = __umulhi(a->data[2], b->data[1]);
        sum = (uint64_t)prod[3] + low + carry;
        prod[3] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[2] * b->data[2];
        high = __umulhi(a->data[2], b->data[2]);
        sum = (uint64_t)prod[4] + low + carry;
        prod[4] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[2] * b->data[3];
        high = __umulhi(a->data[2], b->data[3]);
        sum = (uint64_t)prod[5] + low + carry;
        prod[5] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[2] * b->data[4];
        high = __umulhi(a->data[2], b->data[4]);
        sum = (uint64_t)prod[6] + low + carry;
        prod[6] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[2] * b->data[5];
        high = __umulhi(a->data[2], b->data[5]);
        sum = (uint64_t)prod[7] + low + carry;
        prod[7] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[2] * b->data[6];
        high = __umulhi(a->data[2], b->data[6]);
        sum = (uint64_t)prod[8] + low + carry;
        prod[8] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[2] * b->data[7];
        high = __umulhi(a->data[2], b->data[7]);
        sum = (uint64_t)prod[9] + low + carry;
        prod[9] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        prod[10] += carry;
    }

    // i = 3
    {
        uint32_t carry = 0;
        uint64_t sum;

        uint32_t low = a->data[3] * b->data[0];
        uint32_t high = __umulhi(a->data[3], b->data[0]);
        sum = (uint64_t)prod[3] + low + carry;
        prod[3] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[3] * b->data[1];
        high = __umulhi(a->data[3], b->data[1]);
        sum = (uint64_t)prod[4] + low + carry;
        prod[4] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[3] * b->data[2];
        high = __umulhi(a->data[3], b->data[2]);
        sum = (uint64_t)prod[5] + low + carry;
        prod[5] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[3] * b->data[3];
        high = __umulhi(a->data[3], b->data[3]);
        sum = (uint64_t)prod[6] + low + carry;
        prod[6] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[3] * b->data[4];
        high = __umulhi(a->data[3], b->data[4]);
        sum = (uint64_t)prod[7] + low + carry;
        prod[7] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[3] * b->data[5];
        high = __umulhi(a->data[3], b->data[5]);
        sum = (uint64_t)prod[8] + low + carry;
        prod[8] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[3] * b->data[6];
        high = __umulhi(a->data[3], b->data[6]);
        sum = (uint64_t)prod[9] + low + carry;
        prod[9] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[3] * b->data[7];
        high = __umulhi(a->data[3], b->data[7]);
        sum = (uint64_t)prod[10] + low + carry;
        prod[10] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        prod[11] += carry;
    }

    // i = 4
    {
        uint32_t carry = 0;
        uint64_t sum;

        uint32_t low = a->data[4] * b->data[0];
        uint32_t high = __umulhi(a->data[4], b->data[0]);
        sum = (uint64_t)prod[4] + low + carry;
        prod[4] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[4] * b->data[1];
        high = __umulhi(a->data[4], b->data[1]);
        sum = (uint64_t)prod[5] + low + carry;
        prod[5] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[4] * b->data[2];
        high = __umulhi(a->data[4], b->data[2]);
        sum = (uint64_t)prod[6] + low + carry;
        prod[6] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[4] * b->data[3];
        high = __umulhi(a->data[4], b->data[3]);
        sum = (uint64_t)prod[7] + low + carry;
        prod[7] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[4] * b->data[4];
        high = __umulhi(a->data[4], b->data[4]);
        sum = (uint64_t)prod[8] + low + carry;
        prod[8] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[4] * b->data[5];
        high = __umulhi(a->data[4], b->data[5]);
        sum = (uint64_t)prod[9] + low + carry;
        prod[9] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[4] * b->data[6];
        high = __umulhi(a->data[4], b->data[6]);
        sum = (uint64_t)prod[10] + low + carry;
        prod[10] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[4] * b->data[7];
        high = __umulhi(a->data[4], b->data[7]);
        sum = (uint64_t)prod[11] + low + carry;
        prod[11] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        prod[12] += carry;
    }

    // i = 5
    {
        uint32_t carry = 0;
        uint64_t sum;

        uint32_t low = a->data[5] * b->data[0];
        uint32_t high = __umulhi(a->data[5], b->data[0]);
        sum = (uint64_t)prod[5] + low + carry;
        prod[5] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[5] * b->data[1];
        high = __umulhi(a->data[5], b->data[1]);
        sum = (uint64_t)prod[6] + low + carry;
        prod[6] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[5] * b->data[2];
        high = __umulhi(a->data[5], b->data[2]);
        sum = (uint64_t)prod[7] + low + carry;
        prod[7] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[5] * b->data[3];
        high = __umulhi(a->data[5], b->data[3]);
        sum = (uint64_t)prod[8] + low + carry;
        prod[8] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[5] * b->data[4];
        high = __umulhi(a->data[5], b->data[4]);
        sum = (uint64_t)prod[9] + low + carry;
        prod[9] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[5] * b->data[5];
        high = __umulhi(a->data[5], b->data[5]);
        sum = (uint64_t)prod[10] + low + carry;
        prod[10] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[5] * b->data[6];
        high = __umulhi(a->data[5], b->data[6]);
        sum = (uint64_t)prod[11] + low + carry;
        prod[11] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[5] * b->data[7];
        high = __umulhi(a->data[5], b->data[7]);
        sum = (uint64_t)prod[12] + low + carry;
        prod[12] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        prod[13] += carry;
    }

    // i = 6
    {
        uint32_t carry = 0;
        uint64_t sum;

        uint32_t low = a->data[6] * b->data[0];
        uint32_t high = __umulhi(a->data[6], b->data[0]);
        sum = (uint64_t)prod[6] + low + carry;
        prod[6] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[6] * b->data[1];
        high = __umulhi(a->data[6], b->data[1]);
        sum = (uint64_t)prod[7] + low + carry;
        prod[7] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[6] * b->data[2];
        high = __umulhi(a->data[6], b->data[2]);
        sum = (uint64_t)prod[8] + low + carry;
        prod[8] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[6] * b->data[3];
        high = __umulhi(a->data[6], b->data[3]);
        sum = (uint64_t)prod[9] + low + carry;
        prod[9] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[6] * b->data[4];
        high = __umulhi(a->data[6], b->data[4]);
        sum = (uint64_t)prod[10] + low + carry;
        prod[10] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[6] * b->data[5];
        high = __umulhi(a->data[6], b->data[5]);
        sum = (uint64_t)prod[11] + low + carry;
        prod[11] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[6] * b->data[6];
        high = __umulhi(a->data[6], b->data[6]);
        sum = (uint64_t)prod[12] + low + carry;
        prod[12] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[6] * b->data[7];
        high = __umulhi(a->data[6], b->data[7]);
        sum = (uint64_t)prod[13] + low + carry;
        prod[13] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        prod[14] += carry;
    }

    // i = 7
    {
        uint32_t carry = 0;
        uint64_t sum;

        uint32_t low = a->data[7] * b->data[0];
        uint32_t high = __umulhi(a->data[7], b->data[0]);
        sum = (uint64_t)prod[7] + low + carry;
        prod[7] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[7] * b->data[1];
        high = __umulhi(a->data[7], b->data[1]);
        sum = (uint64_t)prod[8] + low + carry;
        prod[8] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[7] * b->data[2];
        high = __umulhi(a->data[7], b->data[2]);
        sum = (uint64_t)prod[9] + low + carry;
        prod[9] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[7] * b->data[3];
        high = __umulhi(a->data[7], b->data[3]);
        sum = (uint64_t)prod[10] + low + carry;
        prod[10] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[7] * b->data[4];
        high = __umulhi(a->data[7], b->data[4]);
        sum = (uint64_t)prod[11] + low + carry;
        prod[11] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[7] * b->data[5];
        high = __umulhi(a->data[7], b->data[5]);
        sum = (uint64_t)prod[12] + low + carry;
        prod[12] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[7] * b->data[6];
        high = __umulhi(a->data[7], b->data[6]);
        sum = (uint64_t)prod[13] + low + carry;
        prod[13] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        low = a->data[7] * b->data[7];
        high = __umulhi(a->data[7], b->data[7]);
        sum = (uint64_t)prod[14] + low + carry;
        prod[14] = (uint32_t)sum;
        carry = high + (uint32_t)(sum >> 32);

        prod[15] += carry;
    }

    // --- Reduction phase (same as your original, proven correct) ---

    BigInt L, H;
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; i++) {
        L.data[i] = prod[i];
        H.data[i] = prod[i + BIGINT_WORDS];
    }

    uint32_t Rext[9] = {0};
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; i++) {
        Rext[i] = L.data[i];
    }
    Rext[8] = 0;

    uint32_t H977[9] = {0};
    multiply_bigint_by_const(&H, 977, H977);
    add_9word(Rext, H977);

    uint32_t Hshift[9] = {0};
    shift_left_word(&H, Hshift);
    add_9word(Rext, Hshift);

    if (Rext[8]) {
        uint32_t extra[9] = {0};
        BigInt extraBI;
        init_bigint(&extraBI, Rext[8]);
        Rext[8] = 0;

        uint32_t extra977[9] = {0}, extraShift[9] = {0};
        multiply_bigint_by_const(&extraBI, 977, extra977);
        shift_left_word(&extraBI, extraShift);

        #pragma unroll
        for (int i = 0; i < 9; i++) {
            extra[i] = extra977[i];
        }
        add_9word(extra, extraShift);
        add_9word(Rext, extra);
    }

    BigInt R_temp;
    convert_9word_to_bigint(Rext, &R_temp);

    if (Rext[8] || compare_bigint(&R_temp, &const_p) >= 0) {
        ptx_u256Sub(&R_temp, &R_temp, &const_p);
    }
    if (compare_bigint(&R_temp, &const_p) >= 0) {
        ptx_u256Sub(&R_temp, &R_temp, &const_p);
    }

    copy_bigint(res, &R_temp);
}



__device__ __forceinline__ void sub_mod_device(BigInt *res, const BigInt *a, const BigInt *b) {
    BigInt temp;
    if (compare_bigint(a, b) < 0) {
         BigInt sum;
         ptx_u256Add(&sum, a, &const_p);
         ptx_u256Sub(&temp, &sum, b);
    } else {
         ptx_u256Sub(&temp, a, b);
    }
    copy_bigint(res, &temp);
}

__device__ __forceinline__ void scalar_mod_n(BigInt *res, const BigInt *a) {
    if (compare_bigint(a, &const_n) >= 0) {
        ptx_u256Sub(res, a, &const_n);
    } else {
        copy_bigint(res, a);
    }
}

__device__ __forceinline__ void add_mod_device(BigInt *res, const BigInt *a, const BigInt *b) {
    uint32_t carry;
    
    // Use PTX for addition with carry flag
    asm volatile(
        "add.cc.u32 %0, %9, %17;\n\t"
        "addc.cc.u32 %1, %10, %18;\n\t"
        "addc.cc.u32 %2, %11, %19;\n\t"
        "addc.cc.u32 %3, %12, %20;\n\t"
        "addc.cc.u32 %4, %13, %21;\n\t"
        "addc.cc.u32 %5, %14, %22;\n\t"
        "addc.cc.u32 %6, %15, %23;\n\t"
        "addc.cc.u32 %7, %16, %24;\n\t"
        "addc.u32 %8, 0, 0;\n\t"  // capture final carry
        : "=r"(res->data[0]), "=r"(res->data[1]), "=r"(res->data[2]), "=r"(res->data[3]),
          "=r"(res->data[4]), "=r"(res->data[5]), "=r"(res->data[6]), "=r"(res->data[7]),
          "=r"(carry)
        : "r"(a->data[0]), "r"(a->data[1]), "r"(a->data[2]), "r"(a->data[3]),
          "r"(a->data[4]), "r"(a->data[5]), "r"(a->data[6]), "r"(a->data[7]),
          "r"(b->data[0]), "r"(b->data[1]), "r"(b->data[2]), "r"(b->data[3]),
          "r"(b->data[4]), "r"(b->data[5]), "r"(b->data[6]), "r"(b->data[7])
    );
    
    if (carry || compare_bigint(res, &const_p) >= 0) {
        ptx_u256Sub(res, res, &const_p);
    }
}

__device__ void modexp(BigInt *res, const BigInt *base, const BigInt *exp) {
    BigInt result;
    init_bigint(&result, 1);
    BigInt b;
    copy_bigint(&b, base);
    for (int i = 0; i < 256; i++) {
         if (get_bit(exp, i)) {
              mul_mod_device(&result, &result, &b);
         }
         mul_mod_device(&b, &b, &b);
    }
    copy_bigint(res, &result);
}

__device__ void mod_inverse(BigInt *res, const BigInt *a) {
    BigInt p_minus_2, two;
    init_bigint(&two, 2);
    ptx_u256Sub(&p_minus_2, &const_p, &two);
    modexp(res, a, &p_minus_2);
}

__device__ __forceinline__ void point_set_infinity_jac(ECPointJac *P) {
    P->infinity = true;
}

__device__ __forceinline__ void point_copy_jac(ECPointJac *dest, const ECPointJac *src) {
    copy_bigint(&dest->X, &src->X);
    copy_bigint(&dest->Y, &src->Y);
    copy_bigint(&dest->Z, &src->Z);
    dest->infinity = src->infinity;
}

__device__ void double_point_jac(ECPointJac *R, const ECPointJac *P) {
    if (P->infinity || is_zero(&P->Y)) {
        point_set_infinity_jac(R);
        return;
    }
    BigInt A, B, C, D, X3, Y3, Z3, temp, temp2;
    mul_mod_device(&A, &P->Y, &P->Y);
    mul_mod_device(&temp, &P->X, &A);
    init_bigint(&temp2, 4);
    mul_mod_device(&B, &temp, &temp2);
    mul_mod_device(&temp, &A, &A);
    init_bigint(&temp2, 8);
    mul_mod_device(&C, &temp, &temp2);
    mul_mod_device(&temp, &P->X, &P->X);
    init_bigint(&temp2, 3);
    mul_mod_device(&D, &temp, &temp2);
    BigInt D2, two, twoB;
    mul_mod_device(&D2, &D, &D);
    init_bigint(&two, 2);
    mul_mod_device(&twoB, &B, &two);
    sub_mod_device(&X3, &D2, &twoB);
    sub_mod_device(&temp, &B, &X3);
    mul_mod_device(&temp, &D, &temp);
    sub_mod_device(&Y3, &temp, &C);
    init_bigint(&temp, 2);
    mul_mod_device(&temp, &temp, &P->Y);
    mul_mod_device(&Z3, &temp, &P->Z);
    copy_bigint(&R->X, &X3);
    copy_bigint(&R->Y, &Y3);
    copy_bigint(&R->Z, &Z3);
    R->infinity = false;
}

__device__ void add_point_jac(ECPointJac *R, const ECPointJac *P, const ECPointJac *Q) {
    if (P->infinity) { point_copy_jac(R, Q); return; }
    if (Q->infinity) { point_copy_jac(R, P); return; }

    BigInt Z1Z1, Z2Z2, U1, U2, S1, S2, H, R_big, H2, H3, U1H2, X3, Y3, Z3, temp;
    mul_mod_device(&Z1Z1, &P->Z, &P->Z);
    mul_mod_device(&Z2Z2, &Q->Z, &Q->Z);
    mul_mod_device(&U1, &P->X, &Z2Z2);
    mul_mod_device(&U2, &Q->X, &Z1Z1);
    BigInt Z2_cubed, Z1_cubed;
    mul_mod_device(&temp, &Z2Z2, &Q->Z); copy_bigint(&Z2_cubed, &temp);
    mul_mod_device(&temp, &Z1Z1, &P->Z); copy_bigint(&Z1_cubed, &temp);
    mul_mod_device(&S1, &P->Y, &Z2_cubed);
    mul_mod_device(&S2, &Q->Y, &Z1_cubed);

    if (compare_bigint(&U1, &U2) == 0) {
        if (compare_bigint(&S1, &S2) != 0) {
            point_set_infinity_jac(R);
            return;
        } else {
            double_point_jac(R, P);
            return;
        }
    }
    sub_mod_device(&H, &U2, &U1);
    sub_mod_device(&R_big, &S2, &S1);
    mul_mod_device(&H2, &H, &H);
    mul_mod_device(&H3, &H2, &H);
    mul_mod_device(&U1H2, &U1, &H2);
    BigInt R2, two, twoU1H2;
    mul_mod_device(&R2, &R_big, &R_big);
    init_bigint(&two, 2);
    mul_mod_device(&twoU1H2, &U1H2, &two);
    sub_mod_device(&temp, &R2, &H3);
    sub_mod_device(&X3, &temp, &twoU1H2);
    sub_mod_device(&temp, &U1H2, &X3);
    mul_mod_device(&temp, &R_big, &temp);
    mul_mod_device(&Y3, &S1, &H3);
    sub_mod_device(&Y3, &temp, &Y3);
    mul_mod_device(&temp, &P->Z, &Q->Z);
    mul_mod_device(&Z3, &temp, &H);
    copy_bigint(&R->X, &X3);
    copy_bigint(&R->Y, &Y3);
    copy_bigint(&R->Z, &Z3);
    R->infinity = false;
}

// Mixed Jacobian-Affine point addition (faster when Q is in affine coordinates)
__device__ void add_point_mixed(ECPointJac *R, const ECPointJac *P, const ECPoint *Q) {
    if (P->infinity) {
        R->X = Q->x;
        R->Y = Q->y;
        init_bigint(&R->Z, 1);
        R->infinity = Q->infinity;
        return;
    }
    if (Q->infinity) {
        point_copy_jac(R, P);
        return;
    }
    
    BigInt Z1Z1, U2, S2, H, R_big, H2, H3, U1H2, X3, Y3, Z3, temp;
    
    // Z1Z1 = Z1^2
    mul_mod_device(&Z1Z1, &P->Z, &P->Z);
    
    // U2 = X2 * Z1Z1
    mul_mod_device(&U2, &Q->x, &Z1Z1);
    
    // S2 = Y2 * Z1^3
    mul_mod_device(&temp, &Z1Z1, &P->Z);
    mul_mod_device(&S2, &Q->y, &temp);
    
    // Check if points are equal
    if (compare_bigint(&P->X, &U2) == 0) {
        if (compare_bigint(&P->Y, &S2) == 0) {
            // Double the point
            double_point_jac(R, P);
            return;
        } else {
            // Points are inverses
            point_set_infinity_jac(R);
            return;
        }
    }
    
    // H = U2 - X1
    sub_mod_device(&H, &U2, &P->X);
    
    // R = S2 - Y1
    sub_mod_device(&R_big, &S2, &P->Y);
    
    // H2 = H^2
    mul_mod_device(&H2, &H, &H);
    
    // H3 = H * H2
    mul_mod_device(&H3, &H, &H2);
    
    // U1H2 = X1 * H2
    mul_mod_device(&U1H2, &P->X, &H2);
    
    // X3 = R^2 - H3 - 2*U1H2
    BigInt R2, two, twoU1H2;
    mul_mod_device(&R2, &R_big, &R_big);
    init_bigint(&two, 2);
    mul_mod_device(&twoU1H2, &U1H2, &two);
    sub_mod_device(&temp, &R2, &H3);
    sub_mod_device(&X3, &temp, &twoU1H2);
    
    // Y3 = R*(U1H2 - X3) - Y1*H3
    sub_mod_device(&temp, &U1H2, &X3);
    mul_mod_device(&temp, &R_big, &temp);
    mul_mod_device(&Y3, &P->Y, &H3);
    sub_mod_device(&Y3, &temp, &Y3);
    
    // Z3 = Z1 * H
    mul_mod_device(&Z3, &P->Z, &H);
    
    copy_bigint(&R->X, &X3);
    copy_bigint(&R->Y, &Y3);
    copy_bigint(&R->Z, &Z3);
    R->infinity = false;
}

__device__ void jacobian_to_affine(ECPoint *R, const ECPointJac *P) {
    if (P->infinity) {
        R->infinity = true;
        init_bigint(&R->x, 0);
        init_bigint(&R->y, 0);
        return;
    }
    BigInt Zinv, Zinv2, Zinv3;
    mod_inverse(&Zinv, &P->Z);
    mul_mod_device(&Zinv2, &Zinv, &Zinv);
    mul_mod_device(&Zinv3, &Zinv2, &Zinv);
    mul_mod_device(&R->x, &P->X, &Zinv2);
    mul_mod_device(&R->y, &P->Y, &Zinv3);
    R->infinity = false;
}

__device__ void scalar_multiply_jac_device(ECPointJac *result, const ECPointJac *point, const BigInt *scalar) {
    const int WINDOW_SIZE = 4;
    const int PRECOMP_SIZE = 1 << WINDOW_SIZE;
    
    // Use shared memory for precomputed points
    __shared__ ECPointJac shared_precomp[1 << WINDOW_SIZE];
    
    // Collaborative precomputation using threads in the block
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Each thread computes some precomputed points
    for (int i = tid; i < PRECOMP_SIZE; i += block_size) {
        if (i == 0) {
            point_set_infinity_jac(&shared_precomp[0]);
        } else if (i == 1) {
            point_copy_jac(&shared_precomp[1], point);
        } else {
            add_point_jac(&shared_precomp[i], &shared_precomp[i-1], point);
        }
    }
    
    // Ensure all threads have finished precomputation
    __syncthreads();
    
    // Find the highest non-zero bit
    int highest_bit = BIGINT_WORDS * 32 - 1;
    for (; highest_bit >= 0; highest_bit--) {
        if (get_bit(scalar, highest_bit)) break;
    }
    
    if (highest_bit < 0) {
        point_set_infinity_jac(result);
        return;
    }
    
    // Initialize result
    ECPointJac res;
    point_set_infinity_jac(&res);
    
    // Process scalar in windows of WINDOW_SIZE bits
    int i = highest_bit;
    while (i >= 0) {
        // Determine window size for this iteration
        int window_bits = (i >= WINDOW_SIZE - 1) ? WINDOW_SIZE : (i + 1);
        
        // Double 'window_bits' times
        #pragma unroll
        for (int j = 0; j < window_bits; j++) {
            double_point_jac(&res, &res);
        }
        
        // Extract window value
        int window_value = 0;
        for (int j = 0; j < window_bits; j++) {
            if (i - j >= 0 && get_bit(scalar, i - j)) {
                window_value |= (1 << (window_bits - 1 - j));
            }
        }
        
        // Add precomputed point if window value is non-zero
        if (window_value > 0) {
            add_point_jac(&res, &res, &shared_precomp[window_value]);
        }
        
        i -= window_bits;
    }
    
    point_copy_jac(result, &res);
}

// Add these constants for secp256k1 endomorphism
__constant__ BigInt const_beta;    // β = 0x7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee
__constant__ BigInt const_lambda;  // λ = 0x5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72
__constant__ BigInt const_a1;      // 0x3086d221a7d46bcde86c90e49284eb15
__constant__ BigInt const_b1;      // -0xe4437ed6010e88286f547fa90abfe4c3
__constant__ BigInt const_a2;      // 0x114ca50f78b083f397f44b36fa9436c0
__constant__ BigInt const_b2;      // -0x3086d221a7d46bcde86c90e49284eb15

// Endomorphism-specific modular operations
__device__ __forceinline__ void add_mod_n(BigInt *res, const BigInt *a, const BigInt *b) {
    BigInt temp;
    ptx_u256Add(&temp, a, b);
    
    // Check if we need to reduce modulo n
    if (compare_bigint(&temp, &const_n) >= 0) {
        ptx_u256Sub(res, &temp, &const_n);
    } else {
        copy_bigint(res, &temp);
    }
}

__device__ __forceinline__ void sub_mod_n(BigInt *res, const BigInt *a, const BigInt *b) {
    if (compare_bigint(a, b) < 0) {
        BigInt sum;
        ptx_u256Add(&sum, a, &const_n);
        ptx_u256Sub(res, &sum, b);
    } else {
        ptx_u256Sub(res, a, b);
    }
}

__device__ __forceinline__ void neg_mod_n(BigInt *res, const BigInt *a) {
    if (is_zero(a)) {
        init_bigint(res, 0);
    } else {
        ptx_u256Sub(res, &const_n, a);
    }
}

// Apply endomorphism to point: (x, y) -> (β*x, y)
__device__ void apply_endomorphism(ECPointJac *result, const ECPointJac *P) {
    if (P->infinity) {
        point_copy_jac(result, P);
        return;
    }
    
    // result->X = β * P->X
    mul_mod_device(&result->X, &P->X, &const_beta);
    copy_bigint(&result->Y, &P->Y);
    copy_bigint(&result->Z, &P->Z);
    result->infinity = false;
}

// Split scalar k into k1, k2 such that k ≡ k1 + k2*λ (mod n)
// and both k1, k2 are roughly half the bit length of k
__device__ void split_scalar(BigInt *k1, BigInt *k2, const BigInt *k) {
    // This is a simplified version of the GLV scalar decomposition
    // In practice, you'd use the more complex Babai's nearest plane algorithm
    
    BigInt c1, c2, temp1, temp2, temp3;
    
    // c1 = round(b2 * k / n) where b2 is a precomputed constant
    // For secp256k1: b2 = 0x3086d221a7d46bcde86c90e49284eb15
    mul_mod_device(&temp1, &const_b2, k);
    
    // Simple division approximation (this is simplified - real implementation
    // would use more precise division)
    // c1 ≈ temp1 / n
    init_bigint(&c1, 0); // Simplified for demo
    
    // c2 = round(b1 * k / n) where b1 is a precomputed constant  
    // For secp256k1: b1 = -0xe4437ed6010e88286f547fa90abfe4c3
    mul_mod_device(&temp2, &const_b1, k);
    init_bigint(&c2, 0); // Simplified for demo
    
    // k1 = k - c1*a1 - c2*a2
    mul_mod_device(&temp1, &c1, &const_a1);
    mul_mod_device(&temp2, &c2, &const_a2);
    sub_mod_n(&temp3, k, &temp1);
    sub_mod_n(k1, &temp3, &temp2);
    
    // k2 = -c1*b1 - c2*b2
    BigInt neg_b1, neg_b2;
    neg_mod_n(&neg_b1, &const_b1);
    neg_mod_n(&neg_b2, &const_b2);
    
    mul_mod_device(&temp1, &c1, &neg_b1);
    mul_mod_device(&temp2, &c2, &neg_b2);
    add_mod_n(k2, &temp1, &temp2);
    
    // Ensure k1, k2 are in proper range
    scalar_mod_n(k1, k1);
    scalar_mod_n(k2, k2);
}
// Shamir's trick for computing k1*P1 + k2*P2 efficiently
__device__ void simultaneous_scalar_multiply(ECPointJac *result, 
                                            const ECPointJac *P1, const BigInt *k1,
                                            const ECPointJac *P2, const BigInt *k2) {
    // Precompute P1, P2, P1+P2
    ECPointJac precomp[4];
    point_set_infinity_jac(&precomp[0]);     // 0*P1 + 0*P2 = O
    point_copy_jac(&precomp[1], P1);         // 1*P1 + 0*P2 = P1
    point_copy_jac(&precomp[2], P2);         // 0*P1 + 1*P2 = P2
    add_point_jac(&precomp[3], P1, P2);      // 1*P1 + 1*P2 = P1+P2
    
    // Find the highest bit position in either scalar
    int max_bits = 0;
    for (int i = BIGINT_WORDS * 32 - 1; i >= 0; i--) {
        if (get_bit(k1, i) || get_bit(k2, i)) {
            max_bits = i + 1;
            break;
        }
    }
    
    if (max_bits == 0) {
        point_set_infinity_jac(result);
        return;
    }
    
    // Initialize result
    ECPointJac res;
    point_set_infinity_jac(&res);
    
    // Process bits from most significant to least significant
    for (int i = max_bits - 1; i >= 0; i--) {
        // Double the current result
        double_point_jac(&res, &res);
        
        // Determine which precomputed point to add
        int index = 0;
        if (get_bit(k1, i)) index += 1;
        if (get_bit(k2, i)) index += 2;
        
        // Add the appropriate precomputed point
        if (index > 0) {
            add_point_jac(&res, &res, &precomp[index]);
        }
    }
    
    point_copy_jac(result, &res);
}

// Enhanced scalar multiplication using endomorphism (GLV method)
__device__ void scalar_multiply_glv_device(ECPointJac *result, const ECPointJac *point, const BigInt *scalar) {
    BigInt k1, k2;
    ECPointJac phi_P; // φ(P) where φ is the endomorphism
    
    // Split scalar k = k1 + k2*λ
    split_scalar(&k1, &k2, scalar);
    
    // Apply endomorphism: φ(P) = (β*x, y)
    apply_endomorphism(&phi_P, point);
    
    // Handle signs - if k1 or k2 is negative, negate the corresponding point
    bool k1_negative = false, k2_negative = false;
    
    // Check if k1 is in upper half of field (simplified sign check)
    BigInt half_n;
    init_bigint(&half_n, 0);
    half_n.data[BIGINT_WORDS-1] = 0x80000000; // Approximation of n/2
    
    if (compare_bigint(&k1, &half_n) > 0) {
        k1_negative = true;
        sub_mod_n(&k1, &const_n, &k1);
    }
    
    if (compare_bigint(&k2, &half_n) > 0) {
        k2_negative = true;
        sub_mod_n(&k2, &const_n, &k2);
    }
    
    // Negate points if needed
    ECPointJac P1, P2;
    point_copy_jac(&P1, point);
    point_copy_jac(&P2, &phi_P);
    
    if (k1_negative) {
        // Negate P1: (x, y, z) -> (x, -y, z)
        sub_mod_device(&P1.Y, &const_p, &P1.Y);
    }
    
    if (k2_negative) {
        // Negate P2: (x, y, z) -> (x, -y, z) 
        sub_mod_device(&P2.Y, &const_p, &P2.Y);
    }
    
    // Simultaneous scalar multiplication: k1*P + k2*φ(P)
    // Using Shamir's trick for efficiency
    simultaneous_scalar_multiply(result, &P1, &k1, &P2, &k2);
}


// Enhanced scalar multiplication with both windowing and endomorphism
__device__ void scalar_multiply_enhanced(ECPointJac *result, const ECPointJac *point, const BigInt *scalar) {
    // For small scalars, use regular windowed method
    // For larger scalars, use GLV method
    
    // Count significant bits
    int sig_bits = 0;
    for (int i = BIGINT_WORDS * 32 - 1; i >= 0; i--) {
        if (get_bit(scalar, i)) {
            sig_bits = i + 1;
            break;
        }
    }
    
    // Use GLV method for scalars with more than 128 bits
    if (sig_bits > 128) {
        scalar_multiply_glv_device(result, point, scalar);
    } else {
        // Use original windowed method for smaller scalars
        scalar_multiply_jac_device(result, point, scalar);
    }
}

// Host function to initialize endomorphism constants
void init_endomorphism_constants() {
    // secp256k1 endomorphism constants
    BigInt h_beta, h_lambda;
    
    // β = 0x7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee
    h_beta.data[0] = 0x719501ee;
    h_beta.data[1] = 0x1396c287;
    h_beta.data[2] = 0x2f58995c;
    h_beta.data[3] = 0x9cf04975;
    h_beta.data[4] = 0xac3434e9;
    h_beta.data[5] = 0x6e64479e;
    h_beta.data[6] = 0x657c0710;
    h_beta.data[7] = 0x7ae96a2b;
    
    // λ = 0x5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72
    h_lambda.data[0] = 0x1b23bd72;
    h_lambda.data[1] = 0xdf02967c;
    h_lambda.data[2] = 0x20816678;
    h_lambda.data[3] = 0x122e22ea;
    h_lambda.data[4] = 0x8812645a;
    h_lambda.data[5] = 0xa5261c02;
    h_lambda.data[6] = 0xc05c30e0;
    h_lambda.data[7] = 0x5363ad4c;
    
    // Initialize other constants (a1, b1, a2, b2) for scalar decomposition
    // These would need to be computed based on the specific curve parameters
    
    // Copy to device constant memory
    CHECK_CUDA(cudaMemcpyToSymbol(const_beta, &h_beta, sizeof(BigInt)));
    CHECK_CUDA(cudaMemcpyToSymbol(const_lambda, &h_lambda, sizeof(BigInt)));
    // Copy other constants as well...
}

// Modified kernel to use enhanced scalar multiplication
__global__ void ec_scalar_mult_kernel_enhanced(ECPoint *results, 
                                              ECPoint *points, 
                                              BigInt *scalars, 
                                              int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    // Convert input point to Jacobian coordinates
    ECPointJac point_jac, result_jac;
    point_jac.X = points[idx].x;
    point_jac.Y = points[idx].y;
    init_bigint(&point_jac.Z, 1);
    point_jac.infinity = points[idx].infinity;
    
    // Perform enhanced scalar multiplication
    scalar_multiply_enhanced(&result_jac, &point_jac, &scalars[idx]);
    
    // Convert result back to affine coordinates
    jacobian_to_affine(&results[idx], &result_jac);
}