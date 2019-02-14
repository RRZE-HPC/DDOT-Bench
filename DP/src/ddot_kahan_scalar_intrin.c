#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <omp.h>
#include "immintrin.h"

void ddot_kahan_scalar_intrin(
        int N,
        const double* a,
        const double* b,
        double* r)
{
    if (N == 0) {
        (*r) = 0.0;
        return;
    }

    __m128d sum1, c1, sum2, c2, sum3, c3, sum4, c4;
    sum1 = _mm_set1_pd(0.0); sum2 = _mm_set1_pd(0.0);
    sum3 = _mm_set1_pd(0.0); sum4 = _mm_set1_pd(0.0);
    c1 = _mm_set1_pd(0.0); c2 = _mm_set1_pd(0.0);
    c3 = _mm_set1_pd(0.0); c4 = _mm_set1_pd(0.0);

    int i, rem;
    rem = N % 4;

    __m128d prod1, y1, t1, a1, b1;
    __m128d prod2, y2, t2, a2, b2;
    __m128d prod3, y3, t3, a3, b3;
    __m128d prod4, y4, t4, a4, b4;

    /* use four way unrolling */
    for (i=0; i<N-rem; i+=4) {
        /* load 4x4 doubles into four vector registers */
        a1 = _mm_load_sd(&a[i]);
        a2 = _mm_load_sd(&a[i+1]);
        a3 = _mm_load_sd(&a[i+2]);
        a4 = _mm_load_sd(&a[i]+3);

        /* load 4x4 doubles into four vector registers */
        b1 = _mm_load_sd(&b[i]);
        b2 = _mm_load_sd(&b[i+1]);
        b3 = _mm_load_sd(&b[i+2]);
        b4 = _mm_load_sd(&b[i]+3);

        /* multiply components */
        prod1 = _mm_mul_sd(a1, b1);
        prod2 = _mm_mul_sd(a2, b2);
        prod3 = _mm_mul_sd(a3, b3);
        prod4 = _mm_mul_sd(a4, b4);

        y1 = _mm_sub_sd(prod1, c1);
        y2 = _mm_sub_sd(prod2, c2);
        y3 = _mm_sub_sd(prod3, c3);
        y4 = _mm_sub_sd(prod4, c4);

        t1 = _mm_add_sd(sum1, y1);
        t2 = _mm_add_sd(sum2, y2);
        t3 = _mm_add_sd(sum3, y3);
        t4 = _mm_add_sd(sum4, y4);

        c1 = _mm_sub_sd(_mm_sub_sd(t1, sum1), y1);
        c2 = _mm_sub_sd(_mm_sub_sd(t2, sum2), y2);
        c3 = _mm_sub_sd(_mm_sub_sd(t3, sum3), y3);
        c4 = _mm_sub_sd(_mm_sub_sd(t4, sum4), y4);

        sum1 = t1;
        sum2 = t2;
        sum3 = t3;
        sum4 = t4;
    }

    /* reduce four simd vectors to one simd vector using Kahan */
    c1 = _mm_sub_sd(c1, c2);
    c3 = _mm_sub_sd(c3, c4);

    y1 = _mm_sub_sd(sum2, c1);
    y3 = _mm_sub_sd(sum4, c3);
    t1 = _mm_add_sd(sum1, y1);
    t3 = _mm_add_sd(sum3, y3);
    c1 = _mm_sub_sd(_mm_sub_sd(t1, sum1), y1);
    c3 = _mm_sub_sd(_mm_sub_sd(t3, sum3), y3);
    sum1 = t1;
    sum3 = t3;

    c1 = _mm_sub_sd(c1, c3);
    y1 = _mm_sub_sd(sum3, c1);
    t1 = _mm_add_sd(sum1, y1);
    c1 = _mm_sub_sd(_mm_sub_sd(t1, sum1), y1);
    sum1 = t1;

    /* store results of vector register onto stack */
    double sum;
    double c;
    _mm_store_sd(&sum, sum1);
    _mm_store_sd(&c, c1);

    /* perform scalar Kahan sum of loop remainder */
#pragma novector
    for (i=N-rem; i<N; ++i) {
        double prod = a[i]*b[i];
        double y = prod-c;
        double t = sum+y;
        c = (t-sum)-y;
        sum = t;
    }

    (*r) = sum;
}
