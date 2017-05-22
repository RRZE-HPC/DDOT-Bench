#include <math.h>
#include "immintrin.h"

void ddot_kahan_babuska_scalar_intrin(
        int N,
        const double *X,
        const double *Y,
        double *r)
{
    int i, rem;

    rem = N % 4;

    __m128d x1, x2, x3, x4, y1, y2, y3, y4;
    __m128d a1, a2, a3, a4, b1, b2, b3, b4;
    __m128d prod1, prod2, prod3, prod4;
    __m128d sum1, sum2, sum3, sum4;
    __m128d c1, c2, c3, c4;
    sum1 = _mm_set1_pd(0.0f); sum2 = _mm_set1_pd(0.0f);
    sum3 = _mm_set1_pd(0.0f); sum4 = _mm_set1_pd(0.0f);
    c1 = _mm_set1_pd(0.0f); c2 = _mm_set1_pd(0.0f);
    c3 = _mm_set1_pd(0.0f); c4 = _mm_set1_pd(0.0f);

    for (i=0; i<N-rem; i+=4) {
        /* load 4x4 doubles into four vector registers */
        x1 = _mm_load_sd(&X[i]);
        x2 = _mm_load_sd(&X[i+1]);
        x3 = _mm_load_sd(&X[i+2]);
        x4 = _mm_load_sd(&X[i]+3);

        /* load 4x4 doubles into four vector registers */
        y1 = _mm_load_sd(&Y[i]);
        y2 = _mm_load_sd(&Y[i+1]);
        y3 = _mm_load_sd(&Y[i+2]);
        y4 = _mm_load_sd(&Y[i]+3);

        /* multiply components */
        prod1 = _mm_mul_sd(x1, y1);
        prod2 = _mm_mul_sd(x2, y2);
        prod3 = _mm_mul_sd(x3, y3);
        prod4 = _mm_mul_sd(x4, y4);

        /* set a, b */
        __m128d mask1, mask2, mask3, mask4;
        mask1 = _mm_cmp_sd(sum1, prod1, _CMP_GT_OQ);
        mask2 = _mm_cmp_sd(sum2, prod2, _CMP_GT_OQ);
        mask3 = _mm_cmp_sd(sum3, prod3, _CMP_GT_OQ);
        mask4 = _mm_cmp_sd(sum4, prod4, _CMP_GT_OQ);
        a1 = _mm_blendv_pd(prod1, sum1, mask1);
        a2 = _mm_blendv_pd(prod2, sum2, mask2);
        a3 = _mm_blendv_pd(prod3, sum3, mask3);
        a4 = _mm_blendv_pd(prod4, sum4, mask4);
        b1 = _mm_blendv_pd(sum1, prod1, mask1);
        b2 = _mm_blendv_pd(sum2, prod2, mask2);
        b3 = _mm_blendv_pd(sum3, prod3, mask3);
        b4 = _mm_blendv_pd(sum4, prod4, mask4);

        /* add product to sum */
        sum1 = _mm_add_sd(sum1, prod1);
        sum2 = _mm_add_sd(sum2, prod2);
        sum3 = _mm_add_sd(sum3, prod3);
        sum4 = _mm_add_sd(sum4, prod4);

        /* correction term */
        c1 = _mm_add_sd(c1, _mm_add_sd(_mm_sub_sd(a1, sum1), b1));
        c2 = _mm_add_sd(c2, _mm_add_sd(_mm_sub_sd(a2, sum2), b2));
        c3 = _mm_add_sd(c3, _mm_add_sd(_mm_sub_sd(a3, sum3), b3));
        c4 = _mm_add_sd(c4, _mm_add_sd(_mm_sub_sd(a4, sum4), b4));
    }

    /* reduce four simd vectors to one simd vector */

    __m128d mask1, mask2;
    mask1 = _mm_cmp_sd(sum1, sum2, _CMP_GT_OQ);
    mask2 = _mm_cmp_sd(sum3, sum4, _CMP_GT_OQ);
    a1 = _mm_blendv_pd(sum2, sum1, mask1);
    a2 = _mm_blendv_pd(sum4, sum3, mask2);
    b1 = _mm_blendv_pd(sum1, sum2, mask1);
    b2 = _mm_blendv_pd(sum3, sum4, mask2);
    sum1 = _mm_add_sd(sum1, sum2);
    sum2 = _mm_add_sd(sum3, sum4);
    c1 = _mm_add_sd(c1, _mm_add_sd(_mm_sub_sd(a1, sum1), b1));
    c2 = _mm_add_sd(c2, _mm_add_sd(_mm_sub_sd(a2, sum2), b2));

    mask1 = _mm_cmp_sd(sum1, sum2, _CMP_GT_OQ);
    a1 = _mm_blendv_pd(sum2, sum1, mask1);
    b1 = _mm_blendv_pd(sum1, sum2, mask1);
    sum1 = _mm_add_sd(sum1, sum2);
    c1 = _mm_add_sd(c1, _mm_add_sd(_mm_sub_sd(a1, sum1), b1));

    double sum;
    double c;
    _mm_store_sd(&sum, sum1);
    _mm_store_sd(&c, c1);

    /* perform scalar Kahan sum of loop remainer */
#pragma novector
    for (i=N-rem; i<N; ++i) {
        double prod = X[i]*Y[i];
        double a = (fabs(sum) > fabs(prod)) ? sum : prod;
        double b = (fabs(sum) > fabs(prod)) ? prod : sum;
        sum = sum + prod;
        c = c + ((a - sum) + b);
    }

    sum = sum + c;

    (*r) = sum;
}
