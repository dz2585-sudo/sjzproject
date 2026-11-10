/* Exact accelerated rollout for the frozen 25-cell experiment.
 * Python owns resets and the NumPy RNGs. No fast-math or FP contraction.
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

typedef double (*next_double)(void *);
typedef uint32_t (*next_uint32)(void *);

static const int dx[] = {-1, 1, 0, 0, 0, 0, -3, 3, 0, 0};
static const int dy[] = {0, 0, -1, 1, 0, 0, 0, 0, -3, 3};
static const int hx[] = {1, 0, -1, 0};
static const int hy[] = {0, 1, 0, -1};

static int clip(int x) { return x < 0 ? 0 : (x > 24 ? 24 : x); }

static int advance(int *x, int *y, int *h, int speed) {
    int reflections = 0;
    for (int i = 0; i < speed; ++i) {
        if (*x + hx[*h] != clip(*x + hx[*h]) || *y + hy[*h] != clip(*y + hy[*h])) {
            *h = (*h + 2) % 4;
            ++reflections;
        }
        *x += hx[*h];
        *y += hy[*h];
    }
    return reflections;
}

static int index_of(int xc, int yc, int xt, int yt, int h, int speed) {
    return (((xt - xc + 24) * 49 + (yt - yc + 24)) * 4 + h) * 20 + (speed - 1) * 10;
}

static int greedy(const float *row) {
    int best = 0;
    for (int a = 1; a < 10; ++a)
        if (row[a] > row[best]) best = a;
    return best;
}

static int rule(int xc, int yc, int xt, int yt, int h, int speed) {
    if (xc == xt && yc == yt) return 5;
    advance(&xt, &yt, &h, speed);
    int best = 0, best_d = 100, best_move = 100;
    for (int a = 0; a < 10; ++a) {
        if (a == 5) continue;
        int x = clip(xc + dx[a]), y = clip(yc + dy[a]);
        int d = abs(x - xt) + abs(y - yt), move = abs(x - xc) + abs(y - yc);
        if (d < best_d || (d == best_d && move < best_move)) {
            best = a; best_d = d; best_move = move;
        }
    }
    return best;
}

/* NumPy Generator.integers(10), scalar and default dtype: bounded uint32
 * Lemire multiplication with rejection (2^32 modulo 10 = 6).
 * Use NumPy's callback so its cached upper half remains in the original RNG.
 */
static int random_action(void *state, next_uint32 integer) {
    uint64_t value;
    do { value = (uint64_t)integer(state) * 10u; } while ((uint32_t)value < 6u);
    return (int)(value >> 32);
}

API void rollout(float *q, uint32_t *counts, const int32_t *initial,
                 const double *draws, int n, int offset, int mode,
                 void *rng, next_double uniform, next_uint32 integer,
                 double lambda, double beta, double turn_prob, double toggle_prob,
                 int16_t *out) {
    int xc = initial[0], yc = initial[1], xt = initial[2], yt = initial[3];
    int heading = initial[4], speed = initial[5];
    for (int i = 0; i < n; ++i, out += 16, draws += 3) {
        const int pre = abs(xc - xt) + abs(yc - yt);
        const int old_xt = xt, old_yt = yt;
        const int index = index_of(xc, yc, xt, yt, heading, speed);
        int action;
        if (mode == 2) {
            action = rule(xc, yc, xt, yt, heading, speed);
        } else if (mode == 1 && uniform(rng) < 1.0 + (0.05 - 1.0) * fmin((offset + i) / 4000000.0, 1.0)) {
            action = random_action(rng, integer);
        } else {
            action = greedy(q + index);
        }
        xc = clip(xc + dx[action]);
        yc = clip(yc + dy[action]);
        const int mid = abs(xc - xt) + abs(yc - yt);
        const int hit = action == 5 && mid == 0, miss = action == 5 && !hit;
        const int turn = draws[0] < turn_prob, toggle = draws[2] < toggle_prob;
        if (turn) heading = (heading + (draws[1] < 0.5 ? 1 : 3)) % 4;
        if (toggle) speed = 3 - speed;
        const int reflections = advance(&xt, &yt, &heading, speed);
        const int post = abs(xc - xt) + abs(yc - yt);
        if (mode == 1) {
            const double base = -0.05 + 10.0 * hit - 1.0 * miss;
            const double task = base - lambda * post + beta * (post <= 1);
            const double potential = 0.99 * -post + pre;
            const float *next = q + index_of(xc, yc, xt, yt, heading, speed);
            const double target = (task + potential) + 0.99 * (double)next[greedy(next)];
            const int key = index + action;
            /* NumPy float32 scalar += Python float casts the increment first. */
            q[key] += (float)(0.15 * (target - (double)q[key]));
            ++counts[key];
        }
        out[0] = xc; out[1] = yc; out[2] = xt; out[3] = yt; out[4] = heading; out[5] = speed;
        out[6] = action; out[7] = pre; out[8] = mid; out[9] = post;
        out[10] = abs(xt - old_xt) + abs(yt - old_yt);
        out[11] = hit; out[12] = miss; out[13] = reflections; out[14] = turn; out[15] = toggle;
    }
}
