#ifndef TIMER_H
#define TIMER_H

#include <timer_types.h>

#define RDTSC(cpu_c) \
    __asm__ volatile("xor %%eax,%%eax\n\t"           \
            "cpuid\n\t"           \
            "rdtsc\n\t"           \
            "movl %%eax, %0\n\t"  \
            "movl %%edx, %1\n\t"  \
            : "=r" ((cpu_c).int32.lo), "=r" ((cpu_c).int32.hi) \
            : : "%eax","%ebx","%ecx","%edx")

#define RDTSC_CR(cpu_c) \
    __asm__ volatile(   \
            "rdtsc\n\t"           \
            "movl %%eax, %0\n\t"  \
            "movl %%edx, %1\n\t"  \
            : "=r" ((cpu_c).int32.lo), "=r" ((cpu_c).int32.hi) \
            : : "%eax","%ebx","%ecx","%edx")

#define RDTSCP(cpu_c) \
    __asm__ volatile(     \
            "rdtscp\n\t"          \
            "movl %%eax, %0\n\t"  \
            "movl %%edx, %1\n\t"  \
            "cpuid\n\t"           \
            : "=r" ((cpu_c).int32.lo), "=r" ((cpu_c).int32.hi) \
            : : "%eax","%ebx","%ecx","%edx")

#ifdef HAS_RDTSCP
#define RDTSC_STOP(cpu_c) RDTSCP(cpu_c);
#else
#define RDTSC_STOP(cpu_c) RDTSC_CR(cpu_c);
#endif


extern void timer_init( void );
extern double timer_print( TimerData* );
extern uint64_t timer_printCycles( TimerData* );
extern uint64_t timer_getCpuClock( void );
extern uint64_t timer_getBaseline( void );

static inline void timer_start( TimerData* );
static inline void timer_stop ( TimerData* );

void timer_start( TimerData* time )
{
    RDTSC(time->start);
}

void timer_stop( TimerData* time )
{
    RDTSC_STOP(time->stop)
}


#endif /* TIMER_H */
