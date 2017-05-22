void ddot_naive_vec(
        int N,
        const float* x,
        const float* y,
        float* r)
{
    int i;
    float sum = 0.0;

#pragma vector aligned
    for (i=0; i<N; ++i)
        sum += x[i]*y[i];

    (*r) = sum;
}
