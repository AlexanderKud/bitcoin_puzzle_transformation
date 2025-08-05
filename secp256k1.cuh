// main.cu
#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <string.h>

#define BIGINT_WORDS 8

struct BigInt {
    uint32_t data[BIGINT_WORDS];
};

// secp256k1 prime p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
const uint32_t p_host[BIGINT_WORDS] = {
    0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

// Device constant memory for p and inv
__constant__ BigInt p_dev;
__constant__ uint32_t inv_dev;
__host__ __device__ __forceinline__ int compare_bigint(const BigInt *a, const BigInt *b) {
    for (int i = BIGINT_WORDS - 1; i >= 0; i--) {
        if (a->data[i] > b->data[i]) return 1;
        if (a->data[i] < b->data[i]) return -1;
    }
    return 0;
}
// Utility: print BigInt (host)
void print_bigint(const BigInt *a) {
    for (int i = BIGINT_WORDS - 1; i >= 0; i--) {
        printf("%08x", a->data[i]);
    }
    printf("\n");
}

// Compare BigInts a and b: returns -1 if a < b, 0 if equal, 1 if a > b
__host__ __device__ int compare_bigint(const BigInt *a, const BigInt *b) {
    for (int i = BIGINT_WORDS - 1; i >= 0; i--) {
        if (a->data[i] < b->data[i]) return -1;
        if (a->data[i] > b->data[i]) return 1;
    }
    return 0;
}

// Subtract b from a, store in res: assumes a >= b
__host__ __device__ void bigint_sub(BigInt *res, const BigInt *a, const BigInt *b) {
    uint64_t borrow = 0;
    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint64_t diff = (uint64_t)a->data[i] - b->data[i] - borrow;
        res->data[i] = (uint32_t)diff;
        borrow = (diff >> 63) & 1; // borrow if diff < 0
    }
}

// Set bigint to value (only supports 0 or 1 here)
__host__ __device__ void init_bigint(BigInt *a, uint32_t val) {
    for (int i = 0; i < BIGINT_WORDS; i++) a->data[i] = 0;
    a->data[0] = val;
}

// Multiply 256-bit a and b to 512-bit result T (uint32_t[16])
__device__ void mul_256_256(const BigInt *a, const BigInt *b, uint32_t T[16]) {
    for (int i = 0; i < 16; i++) T[i] = 0;
    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint64_t carry = 0;
        uint32_t ai = a->data[i];
        for (int j = 0; j < BIGINT_WORDS; j++) {
            uint64_t tmp = (uint64_t)ai * b->data[j] + T[i + j] + carry;
            T[i + j] = (uint32_t)tmp;
            carry = tmp >> 32;
        }
        T[i + BIGINT_WORDS] = (uint32_t)carry;
    }
}

// Montgomery reduction: T is 512-bit input, res is 256-bit output
__device__ void montgomery_reduce(BigInt *res, uint32_t T[16]) {
    uint32_t m[BIGINT_WORDS];
    uint64_t carry, sum;
    uint32_t temp[16];
    for (int i = 0; i < 16; i++) temp[i] = T[i];

    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint32_t u = temp[i] * inv_dev;
        m[i] = u;
        carry = 0;
        for (int j = 0; j < BIGINT_WORDS; j++) {
            sum = (uint64_t)u * p_dev.data[j] + temp[i + j] + carry;
            temp[i + j] = (uint32_t)sum;
            carry = sum >> 32;
        }
        int idx = i + BIGINT_WORDS;
        uint64_t s = (uint64_t)temp[idx] + carry;
        temp[idx] = (uint32_t)s;
        // carry = s >> 32; // ignored because temp has 16 words
    }

    for (int i = 0; i < BIGINT_WORDS; i++) {
        res->data[i] = temp[i + BIGINT_WORDS];
    }

    if (compare_bigint(res, &p_dev) >= 0) {
        bigint_sub(res, res, &p_dev);
    }
}

// Montgomery multiplication: res = a * b * R^{-1} mod p
__device__ void montgomery_mul(BigInt *res, const BigInt *a, const BigInt *b) {
    uint32_t T[16];
    mul_256_256(a, b, T);
    montgomery_reduce(res, T);
}

// Convert standard integer a to Montgomery domain: res = a * R mod p
// R = 2^{256} mod p, so this is montgomery_mul(a, R^2 mod p)
__device__ void standard_to_montgomery(BigInt *res, const BigInt *a, const BigInt *R2) {
    montgomery_mul(res, a, R2);
}

// Convert Montgomery domain back to standard integer: res = a * 1 mod p
__device__ void montgomery_to_standard(BigInt *res, const BigInt *a) {
    BigInt one;
    init_bigint(&one, 1);
    montgomery_mul(res, a, &one);
}

// Kernel to do montgomery multiplication of two BigInts
__global__ void montgomery_mul_kernel(BigInt *res, const BigInt *a, const BigInt *b) {
    montgomery_mul(res, a, b);
}

// Host function to compute modular inverse of p[0] mod 2^32 (for Montgomery)
uint32_t modinv32(uint32_t p0) {
    uint32_t inv = 1;
    for (int i = 0; i < 5; i++) {
        inv *= 2 - p0 * inv;
    }
    return ~inv + 1;
}

int main() {
    // Initialize p and inv on host
    BigInt p;
    memcpy(p.data, p_host, sizeof(p.data));
    uint32_t inv = modinv32(p.data[0]);

    // Copy p and inv to constant memory
    cudaMemcpyToSymbol(p_dev, &p, sizeof(BigInt));
    cudaMemcpyToSymbol(inv_dev, &inv, sizeof(uint32_t));

    // Example inputs a and b (random 256-bit numbers)
    BigInt a = {0x12345678,0x9abcdef0,0x0fedcba9,0x87654321,0x11111111,0x22222222,0x33333333,0x44444444};
    BigInt b = {0xdeadbeef,0xcafebabe,0xfaceb00c,0xabad1dea,0x55555555,0x66666666,0x77777777,0x88888888};
    BigInt *d_a, *d_b, *d_res;

    cudaMalloc(&d_a, sizeof(BigInt));
    cudaMalloc(&d_b, sizeof(BigInt));
    cudaMalloc(&d_res, sizeof(BigInt));

    cudaMemcpy(d_a, &a, sizeof(BigInt), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(BigInt), cudaMemcpyHostToDevice);

    montgomery_mul_kernel<<<1,1>>>(d_res, d_a, d_b);

    BigInt res;
    cudaMemcpy(&res, d_res, sizeof(BigInt), cudaMemcpyDeviceToHost);

    printf("Montgomery multiplication result:\n");
    print_bigint(&res);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);

    return 0;
}
