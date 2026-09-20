// Microbench for the YuE2 audio tail: clamp stereo float -> int16 interleave +
// WAV write. The GPU does AR/NAR/VAE; this proves the C postprocess path that a
// native encoder would use. Build: gcc -O3 -o postprocess postprocess.c -lm
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
  double seconds = argc > 1 ? atof(argv[1]) : 180.0;
  int sr = 48000;
  long n = (long)(seconds * sr);
  float *l = malloc(n * sizeof(float));
  float *r = malloc(n * sizeof(float));
  int16_t *pcm = malloc(n * 2 * sizeof(int16_t));
  if (!l || !r || !pcm) return 1;
  for (long i = 0; i < n; i++) {
    l[i] = (float)(i % 48000) / 48000.0f * 2.0f - 1.5f;
    r[i] = -(float)(i % 24000) / 24000.0f + 0.8f;
  }
  double t0 = now_s();
  for (long i = 0; i < n; i++) {
    float a = l[i] < -1 ? -1 : (l[i] > 1 ? 1 : l[i]);
    float b = r[i] < -1 ? -1 : (r[i] > 1 ? 1 : r[i]);
    pcm[2 * i] = (int16_t)(a * 32767.0f);
    pcm[2 * i + 1] = (int16_t)(b * 32767.0f);
  }
  double t1 = now_s();
  FILE *f = fopen("/tmp/yue2-bench.wav", "wb");
  if (!f) return 1;
  uint32_t data = (uint32_t)(n * 2 * sizeof(int16_t));
  uint32_t riff = 36 + data;
  uint8_t hdr[44] = {0};
  memcpy(hdr, "RIFF", 4);
  memcpy(hdr + 4, &riff, 4);
  memcpy(hdr + 8, "WAVEfmt ", 8);
  hdr[16] = 16;
  hdr[20] = 1;
  hdr[22] = 2;
  hdr[24] = sr & 0xff;
  hdr[25] = (sr >> 8) & 0xff;
  uint32_t br = sr * 2 * 2;
  memcpy(hdr + 28, &br, 4);
  hdr[32] = 4;
  hdr[34] = 16;
  memcpy(hdr + 36, "data", 4);
  memcpy(hdr + 40, &data, 4);
  fwrite(hdr, 1, 44, f);
  fwrite(pcm, 1, data, f);
  fclose(f);
  double t2 = now_s();
  printf("samples=%ld convert=%.3fs write=%.3fs total=%.3fs (%.1fx realtime)\n", n, t1 - t0, t2 - t1,
         t2 - t0, seconds / (t2 - t0));
  return 0;
}
