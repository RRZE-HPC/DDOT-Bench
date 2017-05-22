__attribute__((optimize("no-tree-vectorize")))
void ddot_naive_scalar_comp(
        int N,
        const double* x,
        const double* y,
        double* r)
{
    double sum = 0.0;

#pragma novector
#pragma unroll(8)
    for (int i=0; i<N; ++i)
        sum += x[i]*y[i];

    (*r) = sum;
}
