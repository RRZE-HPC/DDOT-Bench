#include <gmp.h>
#include <mpfr.h>

void ddot_mpfr(
        int n,
        const float* x,
        const float* y,
        float* r)
{
    mpfr_t mps;
    mpfr_t mpp;
    mpfr_init2 (mps, 113);
    mpfr_init2 (mpp, 113);
    mpfr_set_ui (mps, 0, MPFR_RNDN);

    for (int i=0; i < n; i++)
    {
        mpfr_set_d(mpp, (double)x[i], MPFR_RNDN);
        mpfr_mul_d(mpp, mpp, (double)y[i], MPFR_RNDN);
        mpfr_add(mps, mps, mpp, MPFR_RNDN);
    }

    (*r) = (float)mpfr_get_d( mps, MPFR_RNDN);
    mpfr_clear(mps);
    mpfr_clear(mpp);
}
