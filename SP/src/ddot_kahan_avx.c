#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "immintrin.h"

float ddot_kahan_avx(
        int N,
        const float* a,
        const float* b,
        float* r)
{

    if (N == 0)
        return 0.0f;

    __m256 sum1, c1, sum2, c2, sum3, c3, sum4, c4;
    sum1 = _mm256_set1_ps(0.0); sum2 = _mm256_set1_ps(0.0);
    sum3 = _mm256_set1_ps(0.0); sum4 = _mm256_set1_ps(0.0);
    c1 = _mm256_set1_ps(0.0); c2 = _mm256_set1_ps(0.0);
    c3 = _mm256_set1_ps(0.0); c4 = _mm256_set1_ps(0.0);

    int i, rem;
    rem = N % 32;

    __m256 prod1, y1, t1, a1, b1;
    __m256 prod2, y2, t2, a2, b2;
    __m256 prod3, y3, t3, a3, b3;
    __m256 prod4, y4, t4, a4, b4;

    /* use four way unrolling */
    for (i=0; i<N-rem; i+=32) {
        /* load 4x4 floats into four vector registers */
        a1 = _mm256_loadu_ps(&a[i]);
        a2 = _mm256_loadu_ps(&a[i+8]);
        a3 = _mm256_loadu_ps(&a[i+16]);
        a4 = _mm256_loadu_ps(&a[i]+24);

        /* load 4x4 floats into four vector registers */
        b1 = _mm256_loadu_ps(&b[i]);
        b2 = _mm256_loadu_ps(&b[i+8]);
        b3 = _mm256_loadu_ps(&b[i+16]);
        b4 = _mm256_loadu_ps(&b[i]+24);

        /* multiply components */
        prod1 = _mm256_mul_ps(a1, b1);
        prod2 = _mm256_mul_ps(a2, b2);
        prod3 = _mm256_mul_ps(a3, b3);
        prod4 = _mm256_mul_ps(a4, b4);

        y1 = _mm256_sub_ps(prod1, c1);
        y2 = _mm256_sub_ps(prod2, c2);
        y3 = _mm256_sub_ps(prod3, c3);
        y4 = _mm256_sub_ps(prod4, c4);

        t1 = _mm256_add_ps(sum1, y1);
        t2 = _mm256_add_ps(sum2, y2);
        t3 = _mm256_add_ps(sum3, y3);
        t4 = _mm256_add_ps(sum4, y4);

        c1 = _mm256_sub_ps(_mm256_sub_ps(t1, sum1), y1);
        c2 = _mm256_sub_ps(_mm256_sub_ps(t2, sum2), y2);
        c3 = _mm256_sub_ps(_mm256_sub_ps(t3, sum3), y3);
        c4 = _mm256_sub_ps(_mm256_sub_ps(t4, sum4), y4);

        sum1 = t1;
        sum2 = t2;
        sum3 = t3;
        sum4 = t4;
    }

    /* reduce four simd vectors to one simd vector using Kahan */
    c1 = _mm256_sub_ps(c1, c2);
    c3 = _mm256_sub_ps(c3, c4);

    y1 = _mm256_sub_ps(sum2, c1);
    y3 = _mm256_sub_ps(sum4, c3);
    t1 = _mm256_add_ps(sum1, y1);
    t3 = _mm256_add_ps(sum3, y3);
    c1 = _mm256_sub_ps(_mm256_sub_ps(t1, sum1), y1);
    c3 = _mm256_sub_ps(_mm256_sub_ps(t3, sum3), y3);
    sum1 = t1;
    sum3 = t3;

    c1 = _mm256_sub_ps(c1, c3);
    y1 = _mm256_sub_ps(sum3, c1);
    t1 = _mm256_add_ps(sum1, y1);
    c1 = _mm256_sub_ps(_mm256_sub_ps(t1, sum1), y1);
    sum1 = t1;

    /* store results of vector register onto stack,
     * horizontal reduction in register using AVX hadd
     * won't give us much of a benefit here. */
    float tmp[8];
    float c_tmp[8];
    _mm256_store_ps(&tmp[0], sum1);
    _mm256_store_ps(&c_tmp[0], c1);

    float sum = 0.0;
    float c = c_tmp[0] + c_tmp[1] + c_tmp[2] + c_tmp[3] + c_tmp[4] + c_tmp[5] +
        c_tmp[6] + c_tmp[7];

    /* perform scalar Kahan sum of partial sums */
#pragma novector
    for (i=0; i<8; ++i) {
        float y = tmp[i]-c;
        float t = sum+y;
        c = (t-sum)-y;
        sum = t;
    }

    /* perform scalar Kahan sum of loop remainer */
#pragma novector
    for (i=N-rem; i<N; ++i) {
        float prod = a[i]*b[i];
        float y = prod-c;
        float t = sum+y;
        c = (t-sum)-y;
        sum = t;
    }

    (*r) = sum;
    return c;
}
