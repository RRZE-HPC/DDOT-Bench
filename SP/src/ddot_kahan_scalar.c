#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <omp.h>
#include "immintrin.h"

void ddot_kahan_scalar_intrin(
        int N,
        const float* a,
        const float* b,
        float* r)
{
    if (N == 0) {
        (*r) = 0.0;
        return;
    }

    __m128 sum1, c1, sum2, c2, sum3, c3, sum4, c4;
    sum1 = _mm_set1_ps(0.0); sum2 = _mm_set1_ps(0.0);
    sum3 = _mm_set1_ps(0.0); sum4 = _mm_set1_ps(0.0);
    c1 = _mm_set1_ps(0.0); c2 = _mm_set1_ps(0.0);
    c3 = _mm_set1_ps(0.0); c4 = _mm_set1_ps(0.0);

    int i, rem;
    rem = N % 8;

    __m128 prod1, y1, t1, a1, b1;
    __m128 prod2, y2, t2, a2, b2;
    __m128 prod3, y3, t3, a3, b3;
    __m128 prod4, y4, t4, a4, b4;

    /* use four way unrolling */
    for (i=0; i<N-rem; i+=4) {
        /* load floats */
        a1 = _mm_load_ss(&a[i]);
        a2 = _mm_load_ss(&a[i+1]);
        a3 = _mm_load_ss(&a[i+2]);
        a4 = _mm_load_ss(&a[i]+3);

        /* load floats */
        b1 = _mm_load_ss(&b[i]);
        b2 = _mm_load_ss(&b[i+1]);
        b3 = _mm_load_ss(&b[i+2]);
        b4 = _mm_load_ss(&b[i]+3);

        /* multiply components */
        prod1 = _mm_mul_ss(a1, b1);
        prod2 = _mm_mul_ss(a2, b2);
        prod3 = _mm_mul_ss(a3, b3);
        prod4 = _mm_mul_ss(a4, b4);

        y1 = _mm_sub_ss(prod1, c1);
        y2 = _mm_sub_ss(prod2, c2);
        y3 = _mm_sub_ss(prod3, c3);
        y4 = _mm_sub_ss(prod4, c4);

        t1 = _mm_add_ss(sum1, y1);
        t2 = _mm_add_ss(sum2, y2);
        t3 = _mm_add_ss(sum3, y3);
        t4 = _mm_add_ss(sum4, y4);

        c1 = _mm_sub_ss(_mm_sub_ss(t1, sum1), y1);
        c2 = _mm_sub_ss(_mm_sub_ss(t2, sum2), y2);
        c3 = _mm_sub_ss(_mm_sub_ss(t3, sum3), y3);
        c4 = _mm_sub_ss(_mm_sub_ss(t4, sum4), y4);

        sum1 = t1;
        sum2 = t2;
        sum3 = t3;
        sum4 = t4;
    }

    /* reduce four simd vectors to one simd vector using Kahan */
    c1 = _mm_sub_ss(c1, c2);
    c3 = _mm_sub_ss(c3, c4);

    y1 = _mm_sub_ss(sum2, c1);
    y3 = _mm_sub_ss(sum4, c3);
    t1 = _mm_add_ss(sum1, y1);
    t3 = _mm_add_ss(sum3, y3);
    c1 = _mm_sub_ss(_mm_sub_ss(t1, sum1), y1);
    c3 = _mm_sub_ss(_mm_sub_ss(t3, sum3), y3);
    sum1 = t1;
    sum3 = t3;

    c1 = _mm_sub_ss(c1, c3);
    y1 = _mm_sub_ss(sum3, c1);
    t1 = _mm_add_ss(sum1, y1);
    c1 = _mm_sub_ss(_mm_sub_ss(t1, sum1), y1);
    sum1 = t1;

    /* store results of vector register onto stack */
    float sum;
    float c;
    _mm_store_ss(&sum, sum1);
    _mm_store_ss(&c, c1);

    /* perform scalar Kahan sum of loop remainder */
#pragma novector
    for (i=N-rem; i<N; ++i) {
        float prod = a[i]*b[i];
        float y = prod-c;
        float t = sum+y;
        c = (t-sum)-y;
        sum = t;
    }

    (*r) = sum;
}
