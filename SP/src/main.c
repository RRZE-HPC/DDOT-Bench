#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <math.h>
#include <string.h>

#include <timer.h>

static float printAccuracy(float, float);

extern void ddot_mpfr(int, const float*, const float*, float*);

extern void ddot_naive_scalar(int, const float*, const float*, float*);
extern void ddot_naive_vec(int, const float*, const float*, float*);

extern void ddot_kahan_scalar_intrin(int, const float*, const float*, float*);
extern void ddot_kahan_avx(int, const float*, const float*, float*);

void benchmark(char *version, void(*func)(int, const float *, const float *, float *), int N, const float *x, const float *y) {
    static float ref = 0.0f;
    float result;
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
    /* 'i' was doubled one time to often */
    i = i / 2;

    /* first run should be MPFR to set reference value */
    if (ref == 0.0f) {
        if (strncmp(version, "MPFR", 4) != 0) {
            printf("First version must be MPFR to set reference value!\n");
            exit(EXIT_FAILURE);
        }
        ref = result;
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

static float printAccuracy(float ref, float result)
{
    float acc;

    if (fabs(ref-result) == 0)
    {
        acc = 23;
    }
    else
    {
        acc = log2(fabs(ref)) - log2(fabs(ref-result));
    }


    return acc;
}


static void init_benign(int N, float *x)
{
    /* produce deterministic results */
    srand48(1);

#pragma omp parallel for schedule(static)
    for (int i=0; i<N; ++i)
        x[i] = drand48()/((float)i*(float)i+1.);
}

static void compare(float ref, float res) {
    if (res == ref)
        printf("[OK]\n");
    else {
        printf("[FAIL: Ref: %f Is: %f Ref-Is: %f]\n", ref, res, ref-res);
        exit(EXIT_FAILURE);
    }
}

static void verify_test(float *A, float *B, int N) {
    float ref, res;

    printf("mpfr\t"); ddot_mpfr(N, A, B, &ref); printf("%d\t", N);
    printf("ddot_naive_scalar\t"); ddot_naive_scalar(N, A, B, &res); compare(ref, res);
    printf("ddot_naive_vec\t"); ddot_naive_vec(N, A, B, &res); compare(ref, res);
    printf("ddot_kahan_scalar_intrin\t"); ddot_kahan_scalar_intrin(N, A, B, &res); compare(ref, res);
    printf("ddot_kahan_avx\t"); ddot_kahan_avx(N, A, B, &res); compare(ref, res);
    printf("\n");
}

void verify(void) {
    float *A, *B;
    int N = 4096;

    mem_allocate((void**) &A, 64, N * sizeof(float));
    mem_allocate((void**) &B, 64, N * sizeof(float));

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

void test_precision(int N, float *x, float *y) {
    benchmark("MPFR\t\t\t", ddot_mpfr, N, x, y);
    benchmark("Naive-scalar\t\t", ddot_naive_scalar, N, x, y);
    benchmark("Naive-VEC\t\t", ddot_naive_vec, N, x, y);
    benchmark("Kahan-scalar-intrin\t", ddot_kahan_scalar_intrin, N, x, y);
    benchmark("Kahan-AVX\t\t", ddot_kahan_avx, N, x, y);
}

int main ( int argc, char * argv[] )
{
    float* x;
    float* y;
    int N = atoi(argv[1]);

    if (strncmp("verify", argv[1], 6) == 0) {
        verify();
        exit(EXIT_SUCCESS);
    }

    timer_init();
    printf("**********************************************************\n");
    printf("Length %d\n",N);
    mem_allocate((void**) &x, 64, N * sizeof(float));
    mem_allocate((void**) &y, 64, N * sizeof(float));

    init_benign(N, x);
    init_benign(N, y);
    test_precision(N, x, y);
}

