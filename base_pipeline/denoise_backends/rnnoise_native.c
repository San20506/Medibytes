/* MediBytes RNNoise v0.2 native boundary.
 *
 * This translation unit deliberately exposes only whole-buffer PCM16 processing
 * and the upstream frame-size query.  The Python adapter creates exactly one
 * state for each call and destroys it before returning, so no state crosses
 * samples or conditions.
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include "rnnoise.h"

int medibytes_rnnoise_frame_size(void)
{
    return rnnoise_get_frame_size();
}

int medibytes_rnnoise_process_pcm16(
    const int16_t *input,
    int16_t *output,
    size_t sample_count)
{
    DenoiseState *state;
    float in_frame[480];
    float out_frame[480];
    int frame_size;
    size_t offset;

    if (input == NULL || output == NULL)
        return -1;
    frame_size = rnnoise_get_frame_size();
    if (frame_size <= 0 || frame_size > (int)(sizeof(in_frame) / sizeof(in_frame[0])))
        return -2;

    state = rnnoise_create(NULL);
    if (state == NULL)
        return -3;

    for (offset = 0; offset < sample_count; offset += (size_t)frame_size) {
        size_t remaining = sample_count - offset;
        size_t count = remaining < (size_t)frame_size ? remaining : (size_t)frame_size;
        size_t sample;

        memset(in_frame, 0, sizeof(in_frame));
        for (sample = 0; sample < count; ++sample) {
            int32_t value = input[offset + sample];
            in_frame[sample] = (float)value / 32768.0f;
        }
        rnnoise_process_frame(state, out_frame, in_frame);
        for (sample = 0; sample < count; ++sample) {
            float scaled = out_frame[sample] * 32767.0f;
            int32_t value;
            if (scaled >= 32767.0f)
                value = 32767;
            else if (scaled <= -32768.0f)
                value = -32768;
            else
                value = (int32_t)(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
            output[offset + sample] = (int16_t)value;
        }
    }

    rnnoise_destroy(state);
    return 0;
}
