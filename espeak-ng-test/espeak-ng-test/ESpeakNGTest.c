#include "ESpeakNGTest.h"

int _test_espeak_ng_phoneme_events_cb(short *samples, int num_samples, espeak_EVENT *events) {
    char *out = events->user_data;
    size_t offset;
    (void) samples;
    (void) num_samples;
    for (espeak_EVENT *e = events; e->type != 0; e++) {
        if (e->type == espeakEVENT_PHONEME) {
            if (out[0] == 0) offset = 0;
            else {
                offset = strlen(out);
                out[offset++] = ' ';
            }
            strncpy(out + offset, e->id.string, sizeof(e->id.string));
        }
    }
    return 0;
}
