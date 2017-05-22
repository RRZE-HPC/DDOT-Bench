void ddot_naive_scalar(
        int N,
        const float* x,
        const float* y,
        float* r)
{
    float sum = 0.0;

#pragma novector
    for (int i=0; i<N; ++i)
        sum += x[i]*y[i];

    (*r) = sum;
}
