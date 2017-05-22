#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

#include <timer.h>

#ifndef MIN
#define MIN(x,y) ((x)<(y)?(x):(y))
#endif
#ifndef MAX
#define MAX(x,y) ((x)>(y)?(x):(y))
#endif


static uint64_t baseline = 0ULL;
static uint64_t cpuClock = 0ULL;


static uint64_t
getCpuSpeed(void)
{
    TimerData data;
    TscCounter start;
    TscCounter stop;
    uint64_t result = 0xFFFFFFFFFFFFFFFFULL;
    struct timeval tv1;
    struct timeval tv2;
    struct timezone tzp;
    struct timespec delay = { 0, 800000000 }; /* calibration time: 800 ms */

    for (int i=0; i< 10; i++)
    {
        timer_start(&data);
        timer_stop(&data);
        result = MIN(result,timer_printCycles(&data));
    }

    baseline = result;
    result = 0xFFFFFFFFFFFFFFFFULL;

    for (int i=0; i< 2; i++)
    {
        RDTSC(start);
        gettimeofday( &tv1, &tzp);
        nanosleep( &delay, NULL);
        RDTSC_STOP(stop);
        gettimeofday( &tv2, &tzp);

        result = MIN(result,(stop.int64 - start.int64));
    }

    return (result) * 1000000 /
        (((uint64_t)tv2.tv_sec * 1000000 + tv2.tv_usec) -
         ((uint64_t)tv1.tv_sec * 1000000 + tv1.tv_usec));
}


void timer_init( void )
{
    cpuClock = getCpuSpeed();
}

uint64_t timer_printCycles( TimerData* time )
{
    /* clamp to zero if something goes wrong */
    if ((time->stop.int64-baseline) < time->start.int64)
    {
        return 0ULL;
    }
    else
    {
        return (time->stop.int64 - time->start.int64 - baseline);
    }
}

/* Return time duration in seconds */
double timer_print( TimerData* time )
{
    uint64_t cycles;

    /* clamp to zero if something goes wrong */
   if ((time->stop.int64-baseline) < time->start.int64)
    {
        cycles = 0ULL;
    }
    else
    {
        cycles = time->stop.int64 - time->start.int64 - baseline;
    }

    return  ((double) cycles / (double) cpuClock);
}

uint64_t timer_getCpuClock( void )
{
    return cpuClock;
}

uint64_t timer_getBaseline( void )
{
    return baseline;
}


