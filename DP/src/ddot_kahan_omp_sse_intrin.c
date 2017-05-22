#ifdef __SSE__
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "immintrin.h"

extern double ddot_kahan_sse_intrin(int, const double*, const double*,
        double*);

void ddot_kahan_omp_sse_intrin(int N, const double *a, const double *b, double *r) {
    double *sum;
    double *c;
    int nthreads;

#pragma omp parallel
    {
#pragma omp single
        {
            nthreads = omp_get_num_threads();
            if ((sum = (double *)malloc(nthreads * sizeof(double))) == NULL) {
                perror("malloc");
                exit(EXIT_FAILURE);
            }
            if ((c = (double *)malloc(nthreads * sizeof(double))) == NULL) {
                perror("malloc");
                exit(EXIT_FAILURE);
            }
        }
    }

    if (N < nthreads) {
        ddot_kahan_sse_intrin(N, a, b, r);
        return;
    }

#pragma omp parallel
    {
        int i, id;
        id = omp_get_thread_num();

        /* calculate each threads chunk */
        int alignment = 64 / sizeof(double);
        int gchunk = ((N/alignment)+(nthreads-1))/nthreads;
        gchunk = gchunk * alignment;
        int chunk = gchunk;
        if ((id+1)*chunk > N)
            chunk = N-(id*chunk);
        if (chunk < 0)
            chunk = 0;

        /* each thread sums part of the array */
        c[id] = ddot_kahan_sse_intrin(chunk, a+id*gchunk, b+id*gchunk, &sum[id]);
    } // end #pragma omp parallel

    /* perform scalar Kahan sum of partial sums */
    double scalar_c = 0.0f;
    double scalar_sum = 0.0f;

#pragma novector
    for (int i=0; i<nthreads; ++i) {
        scalar_c = scalar_c + c[i];
        double y = sum[i]-scalar_c;
        double t = scalar_sum+y;
        scalar_c = (t-scalar_sum)-y;
        scalar_sum = t;
    }

    sum[0] = scalar_sum;

    *r = sum[0];
}
#endif
