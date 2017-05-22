void ddot_naive_vec_comp(
        int N,
        const double *x,
        const double *y,
        double* r)
{
    int i;
    double sum = 0.0;

#pragma vector aligned
#pragma unroll(8)
    for (i=0; i<N; ++i)
        sum += x[i]*y[i];

    (*r) = sum;
}
