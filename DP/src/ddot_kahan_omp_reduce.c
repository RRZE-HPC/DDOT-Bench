#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <omp.h>
#include "immintrin.h"

__attribute__((optimize("no-tree-vectorize")))
void ddot_kahan_omp_scalar_comp_reduce(
        int N,
        const double* a,
        const double* b,
        double* r)
{
    int i;
    double sum = 0.0;

#pragma omp parallel for reduction(+:sum) private(i)
#pragma novector
    for (i=0; i<N; ++i) {
        double c = 0.0;
        double prod = a[i]*b[i];
        double y = prod-c;
        double t = sum+y;
        c = (t-sum)-y;
        sum = t;
    }

    (*r) = sum;
}

__attribute__((optimize("no-tree-vectorize")))
void ddot_kahan_omp_scalar_comp_kahan(
        int N,
        const double* a,
        const double* b,
        double* r)
{
    int i, nthreads;
    double *sum_reduce, *c_reduce;

#pragma omp parallel
    {
    double sum = 0.0;
    double c = 0.0;
    int id = omp_get_thread_num();

#pragma omp single
        {
            nthreads = omp_get_num_threads();
            if ((sum_reduce = (double *)malloc(nthreads * sizeof(double))) == NULL) {
                perror("malloc");
                exit(EXIT_FAILURE);
            }
            if ((c_reduce = (double *)malloc(nthreads * sizeof(double))) == NULL) {
                perror("malloc");
                exit(EXIT_FAILURE);
            }
        }

#pragma omp for private(i)
#pragma novector
        for (i=0; i<N; ++i) {
            double prod = a[i]*b[i];
            double y = prod-c;
            double t = sum+y;
            c = (t-sum)-y;
            sum = t;
        }

        sum_reduce[id] = sum;
        c_reduce[id] = c;
    }

    /* perform scalar Kahan sum of partial sums */
    double scalar_c = 0.0f;
    double scalar_sum = 0.0f;

#pragma novector
    for (int i=0; i<nthreads; ++i) {
        scalar_c = scalar_c + c_reduce[i];
        double y = sum_reduce[i]-scalar_c;
        double t = scalar_sum+y;
        scalar_c = (t-scalar_sum)-y;
        scalar_sum = t;
    }

    (*r) = scalar_sum;
}

__attribute__((optimize("no-tree-vectorize")))
void ddot_kahan_omp_scalar_comp_nokahan(
        int N,
        const double* a,
        const double* b,
        double* r)
{
    int i, nthreads;
    double *sum_reduce, *c_reduce;

#pragma omp parallel
    {
    double sum = 0.0;
    double c = 0.0;
    int id = omp_get_thread_num();

#pragma omp single
        {
            nthreads = omp_get_num_threads();
            if ((sum_reduce = (double *)malloc(nthreads * sizeof(double))) == NULL) {
                perror("malloc");
                exit(EXIT_FAILURE);
            }
            if ((c_reduce = (double *)malloc(nthreads * sizeof(double))) == NULL) {
                perror("malloc");
                exit(EXIT_FAILURE);
            }
        }

#pragma omp for private(i)
#pragma novector
        for (i=0; i<N; ++i) {
            double prod = a[i]*b[i];
            double y = prod-c;
            double t = sum+y;
            c = (t-sum)-y;
            sum = t;
        }

        sum_reduce[id] = sum;
        c_reduce[id] = c;
    }

    /* perform scalar Kahan sum of partial sums */
    double scalar_sum = 0.0f;

#pragma novector
    for (int i=0; i<nthreads; ++i) {
        scalar_sum += sum_reduce[i];
    }

    (*r) = scalar_sum;
}
