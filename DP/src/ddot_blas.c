/* modified SUM routine extracted from XBLAS */
void ddot_blas(int N, const double *A, const double *B, double *r)
{
    double head_tmp, tail_tmp;

    head_tmp = tail_tmp = 0.0;

    for (int i=0; i<N; ++i) {
        double prod = A[i]*B[i];

        /* Compute double-double = double-double + double. */
        double e, t1, t2;

        /* Knuth trick. */
        t1 = head_tmp + prod;
        e = t1 - head_tmp;
        t2 = ((prod - e) + (head_tmp - (t1 - e))) + tail_tmp;

        /* The result is t1 + t2, after normalization. */
        head_tmp = t1 + t2;
        tail_tmp = t2 - (head_tmp - t1);
    }
    *r = head_tmp;
}
