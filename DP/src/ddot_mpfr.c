#include <gmp.h>
#include <mpfr.h>

void ddot_mpfr(
        int n,
        const double* x,
        const double* y,
        double* r)
{
    mpfr_t mps;
    mpfr_t mpp;
    mpfr_init2 (mps, 113);
    mpfr_init2 (mpp, 113);
    mpfr_set_ui (mps, 0, MPFR_RNDN);

    for (int i=0; i < n; i++)
    {
        mpfr_set_d(mpp, x[i], MPFR_RNDN);
        mpfr_mul_d(mpp, mpp, y[i], MPFR_RNDN);
        mpfr_add(mps, mps, mpp, MPFR_RNDN);
    }

    (*r) = mpfr_get_d( mps, MPFR_RNDN);
    mpfr_clear(mps);
    mpfr_clear(mpp);
}
