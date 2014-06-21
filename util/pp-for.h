#ifndef PP_FOR
#define PP_FOR

/* A simple preprocessor-expanded FOR loop.  */

/* Helper macros.  */

#define FOR1(i, o, A...) do {enum {i = o}; A} while (0)
#define FOR2(i, o, A...) FOR1(i, o, A); FOR1(i, o + 1, A)
#define FOR4(i, o, A...) FOR2(i, o, A); FOR2(i, o + 2, A)
#define FOR_(i, N, A...) FOR##N(i, 0, A)

/* Duplicate ACTION N times while assigning values
   0, 1, ..., N - 1 to ITER.  */

#define FOR(ITER, N, ACTION...) FOR_(ITER, N, ACTION)

#endif
