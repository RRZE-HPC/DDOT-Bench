#include <math.h>

void ddot_kahan_babuska_vec_comp(
        int N,
        const double *A,
        const double *B,
        double *r)
{
    double sum = 0.0;
    double c = 0.0;

#pragma vector aligned
#pragma simd reduction(+:c,sum)
    for (int i=0; i<N; ++i) {
        double prod = A[i]*B[i];
        double a = (fabs(sum) > fabs(prod)) ? sum : prod;
        double b = (fabs(sum) > fabs(prod)) ? prod : sum;
        sum = sum + prod;
        c = c + ((a - sum) + b);
    }

    sum = sum + c;

    (*r) = sum;
}
