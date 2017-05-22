#include <stdio.h>

__attribute__((optimize("no-tree-vectorize")))
void ddot_kahan_scalar_comp(
        int N,
        const double* a,
        const double* b,
        double* r)
{
    int i;
    double sum = 0.0;
    double c = 0.0;

#pragma novector
    for (i=0; i<N; ++i) {
        double prod = a[i]*b[i];
        double y = prod-c;
        double t = sum+y;
        c = (t-sum)-y;
        sum = t;
    }

    (*r) = sum;
}
