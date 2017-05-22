#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <math.h>
#include <string.h>

#include <timer.h>

static double printAccuracy(double, double);

extern void ddot_mpfr(int, const double*, const double*, double*);
extern void ddot_blas(int, const double*, const double*, double*);

/* Naive versions. High-level Code */
extern void ddot_naive_scalar_comp(int, const double*, const double*, double*);
extern void ddot_naive_vec_comp(int, const double*, const double*, double*);

/* Kahan: Scalar gets intrinsic variant, because compiler performs poorly */
extern void ddot_kahan_scalar_comp(int, const double*, const double*, double*);
extern void ddot_kahan_scalar_intrin(int, const double*, const double*, double*);
extern void ddot_kahan_sse_intrin(int, const double*, const double*, double*);
extern void ddot_kahan_omp_sse_intrin(int, const double*, const double*, double*);
#ifdef __AVX__
extern void ddot_kahan_avx_intrin(int, const double*, const double*, double*);
extern void ddot_kahan_avx2_asm(int, const double*, const double*, double*);
extern void ddot_kahan_omp_avx_intrin(int, const double*, const double*, double*);
extern void ddot_kahan_omp_avx2_asm(int, const double*, const double*, double*);
#endif

/* Different OpenMP reduction variants */
extern void ddot_kahan_omp_scalar_comp_reduce(int, const double*, const double*, double*);
extern void ddot_kahan_omp_scalar_comp_kahan(int, const double*, const double*, double*);
extern void ddot_kahan_omp_scalar_comp_nokahan(int, const double*, const double*, double*);

/* Kahan-Babuska: Scalar gets intrinsic variant, because compiler performs poorly */
extern void ddot_kahan_babuska_scalar_comp(int, const double*, const double*, double*);
extern void ddot_kahan_babuska_vec_comp(int, const double*, const double*, double*);
extern void ddot_kahan_babuska_scalar_intrin(int, const double*, const double*, double*);
extern void ddot_kahan_babuska_sse_intrin(int, const double*, const double*, double*);
#ifdef __AVX__
extern void ddot_kahan_babuska_avx_intrin(int, const double*, const double*, double*);
#endif

void benchmark(char *version, void(*func)(int, const double *, const double *, double *), int N, const double *x, const double *y) {
    static int refrun = 0;
    static double ref = 0.0f;
    double result;
    int i, j;
    double runtime = 0.0f;

    /* run benchmark and increase number of iterations until runtime > 0.1s */
    for (i=1; runtime < 0.1f; i=i*2) {
        TimerData tdata;
        timer_start(&tdata);
        for (j=0; j<i; ++j)
            func(N, x, y, &result);
        timer_stop(&tdata);
        runtime = timer_print(&tdata);
    }
    /* 'i' was doubled one time too often */
    i = i / 2;

    /* first run should be MPFR to set reference value */
    if (refrun == 0) {
        if (strncmp(version, "Reference", 9) != 0) {
            printf("First version must be MPFR to set reference value!\n");
            exit(EXIT_FAILURE);
        }
        ref = result;
        refrun = 1;
    }

    /* output results */
    printf("%s", version);
    printf("Result %20.15f\t\t", result);
    printf("Perf %f Elements/s\t",((double)N * (double)i)/runtime);
    printf("Accuracy %g bits\t", printAccuracy(ref, result));
    printf("%d iterations\n", i);
}


static void mem_allocate(
    void** ptr,
    int alignment,
    uint64_t size)
{
    int errorCode;

    errorCode =  posix_memalign(ptr, alignment, size);

    if (errorCode)
    {
        if (errorCode == EINVAL)
        {
            fprintf(stderr,
                    "Alignment parameter is not a power of two\n");
            exit(EXIT_FAILURE);
        }
        if (errorCode == ENOMEM)
        {
            fprintf(stderr,
                    "Insufficient memory to fulfill the request\n");
            exit(EXIT_FAILURE);
        }
    }

    if ((*ptr) == NULL)
    {
        fprintf(stderr, "posix_memalign failed!\n");
        exit(EXIT_FAILURE);
    }
}

static double printAccuracy(double ref, double result)
{
    double acc;

    if (fabs(ref-result) == 0)
    {
        acc = 53;
    }
    else
    {
        acc = log2(fabs(ref)) - log2(fabs(ref-result));
    }


    return acc;
}


static void init_benign(int N, double *x)
{
    /* produce deterministic results */
    srand48(1);

#pragma omp parallel for schedule(static)
    for (int i=0; i<N; ++i)
        x[i] = drand48()/((double)i*(double)i+1.);
}

static void compare(double ref, double res) {
    if (res == ref)
        printf("[OK]\n");
    else {
        printf("[FAIL: Ref: %f Is: %f Ref-Is: %f]\n", ref, res, ref-res);
        exit(EXIT_FAILURE);
    }
}

static void verify_test(double *A, double *B, int N) {
    double ref, res;

    ddot_mpfr(N, A, B, &ref); printf("N: %d\t", N);
    printf("ddot_blas\t"); ddot_blas(N, A, B, &res); compare(ref, res);

    /* Naive */
    printf("ddot_naive_scalar_comp\t"); ddot_naive_scalar_comp(N, A, B, &res); compare(ref, res);
    printf("ddot_naive_vec_comp\t"); ddot_naive_vec_comp(N, A, B, &res); compare(ref, res);

    /* Kahan */
    printf("ddot_kahan_scalar_comp\t"); ddot_kahan_scalar_comp(N, A, B, &res); compare(ref, res);
    printf("ddot_kahan_scalar_intrin\t"); ddot_kahan_scalar_intrin(N, A, B, &res); compare(ref, res);
    printf("ddot_kahan_sse_intrin\t"); ddot_kahan_sse_intrin(N, A, B, &res); compare(ref, res);
    printf("ddot_kahan_omp_sse_intrin\t"); ddot_kahan_omp_sse_intrin(N, A, B, &res); compare(ref, res);
#ifdef __AVX__
    printf("ddot_kahan_avx_intrin\t"); ddot_kahan_avx_intrin(N, A, B, &res); compare(ref, res);
    printf("ddot_kahan_avx2_asm\t"); ddot_kahan_avx2_asm(N, A, B, &res); compare(ref, res);
    printf("ddot_kahan_omp_avx_intrin\t"); ddot_kahan_omp_avx_intrin(N, A, B, &res); compare(ref, res);
    printf("ddot_kahan_omp_avx2_asm\t"); ddot_kahan_omp_avx2_asm(N, A, B, &res); compare(ref, res);
#endif

    /* OpenMP Reduction Variants */
    printf("ddot_kahan_omp_scalar_comp_reduce\t"); ddot_kahan_omp_scalar_comp_reduce(N, A, B, &res); compare(ref, res);
    printf("ddot_kahan_omp_scalar_comp_kahan\t"); ddot_kahan_omp_scalar_comp_kahan(N, A, B, &res); compare(ref, res);
    printf("ddot_kahan_omp_scalar_comp_nokahan\t"); ddot_kahan_omp_scalar_comp_nokahan(N, A, B, &res); compare(ref, res);


    /* Kahan-Babuska */
    printf("ddot_kahan_babuska_scalar_comp\t"); ddot_kahan_babuska_scalar_comp(N, A, B, &res); compare(ref, res);
    printf("ddot_kahan_babuska_vec_comp\t"); ddot_kahan_babuska_vec_comp(N, A, B, &res); compare(ref, res);
    printf("ddot_kahan_babuska_scalar_intrin\t"); ddot_kahan_babuska_scalar_intrin(N, A, B, &res); compare(ref, res);
    printf("ddot_kahan_babuska_sse_intrin\t"); ddot_kahan_babuska_sse_intrin(N, A, B, &res); compare(ref, res);
#ifdef __AVX__
    printf("ddot_kahan_babuska_avx_intrin\t"); ddot_kahan_babuska_avx_intrin(N, A, B, &res); compare(ref, res);
#endif
    printf("\n");
}

void verify(void) {
    double *A, *B;
    int N = 4096;

    mem_allocate((void**) &A, 64, N * sizeof(double));
    mem_allocate((void**) &B, 64, N * sizeof(double));

    for (int i=0; i<4096; ++i) {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }

    verify_test(A, B, 4096); /* power of two */
    verify_test(A, B, 2); /* small value, smaller than AVX */
    verify_test(A, B, 11); /* small value, larger than AVX */
    verify_test(A, B, 269); /* medium value */
    verify_test(A, B, 3571); /* prime */

    for (int i=0; i<4096; ++i) {
        A[i] = 1.0;
        B[i] = 2.0;
    }

    verify_test(A, B, 2); /* small value, smaller than AVX */
    verify_test(A, B, 11); /* small value, larger than AVX */
    verify_test(A, B, 512); /* medium value */
    verify_test(A, B, 3000); /* medium value */
}

void test_precision(int N, double *x, double *y) {
    benchmark("Reference\t\t\t", ddot_mpfr, N, x, y);
    benchmark("ddot_blas\t\t\t", ddot_blas, N, x, y);
    /* Naive */
    benchmark("ddot_naive_scalar_comp\t\t", ddot_naive_scalar_comp, N, x, y);
    benchmark("ddot_naive_vec_comp\t\t", ddot_naive_vec_comp, N, x, y);
    /* Kahan */
    benchmark("ddot_kahan_scalar_comp\t\t", ddot_kahan_scalar_comp, N, x, y);
    benchmark("ddot_kahan_scalar_intrin\t\t", ddot_kahan_scalar_intrin, N, x, y);
    benchmark("ddot_kahan_sse_intrin\t\t", ddot_kahan_sse_intrin, N, x, y);
    benchmark("ddot_kahan_omp_sse_intrin\t\t", ddot_kahan_omp_sse_intrin, N, x, y);
#ifdef __AVX__
    benchmark("ddot_kahan_avx_intrin\t\t", ddot_kahan_avx_intrin, N, x, y);
    benchmark("ddot_kahan_avx2_asm\t\t", ddot_kahan_avx2_asm, N, x, y);
    benchmark("ddot_kahan_omp_avx_intrin\t\t", ddot_kahan_omp_avx_intrin, N, x, y);
    benchmark("ddot_kahan_omp_avx2_asm\t\t", ddot_kahan_omp_avx2_asm, N, x, y);
#endif
    /* Different OpenMP reduction variants */
    benchmark("ddot_kahan_omp_scalar_comp_reduce\t\t", ddot_kahan_omp_scalar_comp_reduce, N, x, y);
    benchmark("ddot_kahan_omp_scalar_comp_kahan\t\t", ddot_kahan_omp_scalar_comp_kahan, N, x, y);
    benchmark("ddot_kahan_omp_scalar_comp_nokahan\t\t", ddot_kahan_omp_scalar_comp_nokahan, N, x, y);
    /* Kahan-Babuska */
    benchmark("ddot_kahan_babuska_scalar_comp\t\t", ddot_kahan_babuska_scalar_comp, N, x, y);
    benchmark("ddot_kahan_babuska_vec_comp\t\t", ddot_kahan_babuska_vec_comp, N, x, y);
    benchmark("ddot_kahan_babuska_scalar_intrin\t\t", ddot_kahan_babuska_scalar_intrin, N, x, y);
    benchmark("ddot_kahan_babuska_sse_intrin\t\t", ddot_kahan_babuska_sse_intrin, N, x, y);
    benchmark("ddot_kahan_babuska_avx_intrin\t\t", ddot_kahan_babuska_avx_intrin, N, x, y);

}

int main ( int argc, char * argv[] )
{
    if (argc < 2) {
        printf("usage: %s <elements per array>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    double* x;
    double* y;
    int N = atoi(argv[1]);

    if (strncmp("verify", argv[1], 6) == 0) {
        verify();
        exit(EXIT_SUCCESS);
    }

    timer_init();
    printf("**********************************************************\n");
    printf("Length %d\n",N);
    mem_allocate((void**) &x, 64, N * sizeof(double));
    mem_allocate((void**) &y, 64, N * sizeof(double));

    init_benign(N, x);
    init_benign(N, y);
    test_precision(N, x, y);
}

