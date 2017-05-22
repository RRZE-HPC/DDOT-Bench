#ifndef TIMER_TYPES_H
#define TIMER_TYPES_H

#include <stdint.h>

typedef union
{
    uint64_t int64;
    struct {uint32_t lo, hi;} int32;
} TscCounter;

typedef struct {
    TscCounter start;
    TscCounter stop;
} TimerData;


#endif /*TIMER_TYPES_H*/
