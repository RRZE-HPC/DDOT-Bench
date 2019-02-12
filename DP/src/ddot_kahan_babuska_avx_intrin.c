#ifdef __AVX__
#include <math.h>
#include "immintrin.h"

void ddot_kahan_babuska_avx_intrin(
        int N,
        const double *X,
        const double *Y,
        double *r)
{
    int i, rem;

    rem = N % 16;

    __m256d x1, x2, x3, x4, y1, y2, y3, y4;
    __m256d a1, a2, a3, a4, b1, b2, b3, b4;
    __m256d prod1, prod2, prod3, prod4;
    __m256d sum1, sum2, sum3, sum4;
    __m256d c1, c2, c3, c4;
    sum1 = _mm256_set1_pd(0.0f); sum2 = _mm256_set1_pd(0.0f);
    sum3 = _mm256_set1_pd(0.0f); sum4 = _mm256_set1_pd(0.0f);
    c1 = _mm256_set1_pd(0.0f); c2 = _mm256_set1_pd(0.0f);
    c3 = _mm256_set1_pd(0.0f); c4 = _mm256_set1_pd(0.0f);

    for (i=0; i<N-rem; i+=16) {
        /* load 4x4 doubles into four vector registers */
        x1 = _mm256_load_pd(&X[i]);
        x2 = _mm256_load_pd(&X[i+4]);
        x3 = _mm256_load_pd(&X[i+8]);
        x4 = _mm256_load_pd(&X[i]+12);

        /* load 4x4 doubles into four vector registers */
        y1 = _mm256_load_pd(&Y[i]);
        y2 = _mm256_load_pd(&Y[i+4]);
        y3 = _mm256_load_pd(&Y[i+8]);
        y4 = _mm256_load_pd(&Y[i]+12);

        /* multiply components */
        prod1 = _mm256_mul_pd(x1, y1);
        prod2 = _mm256_mul_pd(x2, y2);
        prod3 = _mm256_mul_pd(x3, y3);
        prod4 = _mm256_mul_pd(x4, y4);

        /* set a, b */
        __m256d mask1, mask2, mask3, mask4;
        mask1 = _mm256_cmp_pd(sum1, prod1, _CMP_GT_OQ);
        mask2 = _mm256_cmp_pd(sum2, prod2, _CMP_GT_OQ);
        mask3 = _mm256_cmp_pd(sum3, prod3, _CMP_GT_OQ);
        mask4 = _mm256_cmp_pd(sum4, prod4, _CMP_GT_OQ);
        a1 = _mm256_blendv_pd(prod1, sum1, mask1);
        a2 = _mm256_blendv_pd(prod2, sum2, mask2);
        a3 = _mm256_blendv_pd(prod3, sum3, mask3);
        a4 = _mm256_blendv_pd(prod4, sum4, mask4);
        b1 = _mm256_blendv_pd(sum1, prod1, mask1);
        b2 = _mm256_blendv_pd(sum2, prod2, mask2);
        b3 = _mm256_blendv_pd(sum3, prod3, mask3);
        b4 = _mm256_blendv_pd(sum4, prod4, mask4);

        /* add product to sum */
        sum1 = _mm256_add_pd(sum1, prod1);
        sum2 = _mm256_add_pd(sum2, prod2);
        sum3 = _mm256_add_pd(sum3, prod3);
        sum4 = _mm256_add_pd(sum4, prod4);

        /* correction term */
        c1 = _mm256_add_pd(c1, _mm256_add_pd(_mm256_sub_pd(a1, sum1), b1));
        c2 = _mm256_add_pd(c2, _mm256_add_pd(_mm256_sub_pd(a2, sum2), b2));
        c3 = _mm256_add_pd(c3, _mm256_add_pd(_mm256_sub_pd(a3, sum3), b3));
        c4 = _mm256_add_pd(c4, _mm256_add_pd(_mm256_sub_pd(a4, sum4), b4));
    }

    /* reduce four simd vectors to one simd vector */

    __m256d mask1, mask2;
    mask1 = _mm256_cmp_pd(sum1, sum2, _CMP_GT_OQ);
    mask2 = _mm256_cmp_pd(sum3, sum4, _CMP_GT_OQ);
    a1 = _mm256_blendv_pd(sum2, sum1, mask1);
    a2 = _mm256_blendv_pd(sum4, sum3, mask2);
    b1 = _mm256_blendv_pd(sum1, sum2, mask1);
    b2 = _mm256_blendv_pd(sum3, sum4, mask2);
    sum1 = _mm256_add_pd(sum1, sum2);
    sum2 = _mm256_add_pd(sum3, sum4);
    c1 = _mm256_add_pd(c1, _mm256_add_pd(_mm256_sub_pd(a1, sum1), b1));
    c2 = _mm256_add_pd(c2, _mm256_add_pd(_mm256_sub_pd(a2, sum2), b2));

    mask1 = _mm256_cmp_pd(sum1, sum2, _CMP_GT_OQ);
    a1 = _mm256_blendv_pd(sum2, sum1, mask1);
    b1 = _mm256_blendv_pd(sum1, sum2, mask1);
    sum1 = _mm256_add_pd(sum1, sum2);
    c1 = _mm256_add_pd(c1, _mm256_add_pd(_mm256_sub_pd(a1, sum1), b1));

    double tmp[4];
    double c_tmp[4];
    _mm256_store_pd(&tmp[0], sum1);
    _mm256_store_pd(&c_tmp[0], c1);

    double sum = 0.0;
    double c = c_tmp[0] + c_tmp[1] + c_tmp[2] + c_tmp[3];

    /* perform scalar Kahan sum of partial sums */
#pragma novector
    for (i=0; i<4; ++i) {
        double a = (fabs(sum) > fabs(tmp[i])) ? sum : tmp[i];
        double b = (fabs(sum) > fabs(tmp[i])) ? tmp[i] : sum;
        sum = sum + tmp[i];
        c = c + ((a - sum) + b);
    }

    /* perform scalar Kahan sum of loop remainder */
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
#endif
