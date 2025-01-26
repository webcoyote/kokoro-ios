#ifndef ESpeakNGTestFunc
#define ESpeakNGTestFunc

#include <stdio.h>
#include <ESpeakNG/espeak_ng.h>
#include <string.h>

int _test_espeak_ng_phoneme_events_cb(short *samples, int num_samples, espeak_EVENT *events);

#endif
