#ifndef FASTAINV_DROPTOL_TUNER_H
#define FASTAINV_DROPTOL_TUNER_H

class DroptolTuner
{
  enum {TEST_M1i, TEST_M1, TEST_M2} phase;

  static const double step_ratio = .3819660112501051518;
  double left_midpoint(double lo, double hi)
  {
    return lo + (hi - lo) * step_ratio;
  }
  double other_midpoint(double lo, double hi, double mid)
  {
    return lo + (hi - mid);
  }

  double droptol_lo, droptol_hi, droptol_m1, droptol_m2, droptol_min;
  double solvetime_m1, solvetime_m2, time_min;
  int iters;

public:

  DroptolTuner(double lo = -10, double hi = -3):
    phase(TEST_M1i), droptol_lo(lo), droptol_hi(hi),
    droptol_m1(left_midpoint(lo, hi)),
    droptol_m2(other_midpoint(lo, hi, droptol_m1)),
    iters(0)
  {}

  double getDroptol()
  {
    return exp2(phase == TEST_M2 ? droptol_m2 : droptol_m1);
  }
  double getBestDroptol()
  {
    return exp2(droptol_min);
  }

  void noteTime(double time)
  {
    switch (phase)
      {
	case TEST_M1i:
	  time_min = time;
	  droptol_min = droptol_m1;
	  solvetime_m1 = time;
	  phase = TEST_M2;
	  return;
	case TEST_M2:
	  if (time_min > time)
	    {
	      time_min = time;
	      droptol_min = droptol_m2;
	    }
	  solvetime_m2 = time;
	  break;
	case TEST_M1:
	  if (time_min > time)
	    {
	      time_min = time;
	      droptol_min = droptol_m1;
	    }
	  solvetime_m1 = time;
	  break;
      }
    if (solvetime_m1 < solvetime_m2)
      {
	droptol_hi = droptol_m2;
	droptol_m2 = droptol_m1;
	droptol_m1 = other_midpoint(droptol_lo, droptol_hi, droptol_m2);
	solvetime_m2 = solvetime_m1;
	phase = TEST_M1;
      }
    else
      {
	droptol_lo = droptol_m1;
	droptol_m1 = droptol_m2;
	droptol_m2 = other_midpoint(droptol_lo, droptol_hi, droptol_m1);
	solvetime_m1 = solvetime_m2;
	phase = TEST_M2;
      }
    iters++;
  }

  void restart()
  {
    phase = TEST_M1i;
    droptol_lo = droptol_m1 - 1;
    droptol_hi = droptol_m1 + 1;
    droptol_m1 = left_midpoint(droptol_lo, droptol_hi);
    droptol_m2 = other_midpoint(droptol_lo, droptol_hi, droptol_m1);
    iters = 0;
  }
  void resetPhase()
  {
    phase = TEST_M1i;
  }
  int iterations()
  {
    return iters;
  }
};

#endif
