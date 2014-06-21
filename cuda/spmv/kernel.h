template<CLASS_T class E, unsigned N, unsigned S, unsigned W, bool var_height, bool diags>
static __global__ void
KERNELNAME
	   (const unsigned *sliceptr,
	    const unsigned *cols,
	    const E *elms,
	    const T *x,
	    T *y)
{
  extern volatile __shared__ double reduce_cols_[];
  volatile T *reduce_cols = (volatile T *) reduce_cols_;

  unsigned slice_ptr_elts_per_slice = 1;
  if (W > 1)
    slice_ptr_elts_per_slice++;
  if (var_height)
    slice_ptr_elts_per_slice++;
  if (diags)
    slice_ptr_elts_per_slice += 2;
  unsigned slice_p0 = slice_ptr_elts_per_slice * blockIdx.x;
  unsigned slice_e0 = fetch<unsigned>(SLICEARG, slice_p0);
  unsigned slice_e1 = fetch<unsigned>(SLICEARG, slice_p0 + slice_ptr_elts_per_slice);
  unsigned slice_y0;
  unsigned slice_sz;
  unsigned slice_c0;
  unsigned slice_c1;
  unsigned slice_d0;
  unsigned slice_cb;
  if (var_height)
    {
      unsigned slice_y1;
      slice_y0 = fetch<unsigned>(SLICEARG, slice_p0 + 1);
      slice_y1 = fetch<unsigned>(SLICEARG, slice_p0 + 1 + slice_ptr_elts_per_slice);
      slice_sz = slice_y1 - slice_y0;
    }
  else
    {
      slice_y0 = S * blockIdx.x;
      slice_sz = S;
    }
  if (diags)
    {
      slice_c0 = fetch<unsigned>(SLICEARG, slice_p0 + var_height + 1);
      slice_d0 = fetch<unsigned>(SLICEARG, slice_p0 + var_height + 2);
      slice_c1 = fetch<unsigned>(SLICEARG, slice_p0 + var_height + 1 + slice_ptr_elts_per_slice);
    }
  else if (W > 1)
    {
      slice_c0 = fetch<unsigned>(SLICEARG, slice_p0 + var_height + 1);
      slice_c1 = fetch<unsigned>(SLICEARG, slice_p0 + var_height + 1 + slice_ptr_elts_per_slice);
      unsigned nse = (W * (slice_c1 - slice_c0) - (slice_e1 - slice_e0));
      nse /= (W > 1) ? (W - 1) : 1;
      slice_cb = slice_c1;
      slice_c1 = slice_c0 + nse;
      slice_e1 = slice_e0 + nse;
    }
  else
    {
      slice_c0 = slice_e0;
      slice_c1 = slice_e1;
    }

  typedef shortvec<T, H> TK;
  typedef shortvec<E, H> EK;
  typedef shortvec<T, W> TW;
  typedef shortvec<unsigned, H> UK;
  unsigned i, j;
  TK t;
  FOR(i, H, t[i] = 0;);
  for (i = slice_e0 + threadIdx.x * H,
       j = slice_c0 + threadIdx.x * H;
       j < slice_c1;
       i += N * H,
       j += N * H)
    {
      UK ii = *(const UK*)&cols[j];
      EK e  = *(const EK*)&elms[i];
      FOR(i, H, t[i] += e[i] * fetch<T>(XARG, ii[i]););
    }
  if (diags)
    {
      unsigned slice_adj = (threadIdx.x * H & (slice_sz - 1)) - slice_sz;
      for (j = slice_d0 + (j - slice_c1) / slice_sz;
	   i < slice_e1;
	   i += N * H,
	   j += N * H / slice_sz)
	{
	  unsigned ii = fetch<unsigned>(SLICEARG, j) + slice_adj;
	  EK e  = *(const EK*)&elms[i];
	  FOR(i, H, t[i] += e[i] * fetch<T>(XARG, ii+i););
	}
    }
  else if (W > 1)
    {
      for (i = (slice_e1 + ((i - slice_e1) & (slice_sz - 1))
		+ ((i - slice_e1) & ~(slice_sz - 1)) * W);
	   j < slice_cb; i += W * N * H, j += N * H)
	{
	  UK ii = *(const UK*)&cols[j];
	  EK e0, e1, e2, e3;
	  e0 = *(const EK*)&elms[i];
	  e1 = *(const EK*)&elms[i + slice_sz];
	  if (W > 2)
	    {
	      e2 = *(const EK*)&elms[i + slice_sz * 2];
	      e3 = *(const EK*)&elms[i + slice_sz * 3];
	    }
	  TW wx;
	  FOR(i, H,
	    {
	      wx = fetch<TW::cuname>(XARGW, ii[i]);
	      t[i] += e0[i] * wx[0] + e1[i] * wx[1];
	      if (W > 2)
	        {
		  t[i] += e2[i] * wx[2] + e3[i] * wx[3];
		}
	    });
	}
    }
  if (var_height)
    {
      reduce_cols_var<T> r;
      FOR(i, H, t[i] = r(reduce_cols, t[i], N, slice_sz / H););
    }
  else
    {
      reduce_cols_cond<T, N, S / H> r;
      FOR(i, H, t[i] = r(reduce_cols, t[i]););
    }
  if (threadIdx.x < slice_sz / H)
    {
      TK *py = (TK *)&y[slice_y0 + threadIdx.x * H];
      *py = t;
    }
}

template<CLASS_T class E, unsigned N, unsigned S, unsigned W, bool V, bool D>
struct spmv_kernel<USE_TEX_SLICEPTR, USE_TEX_X, T, E, N, S, H, W, V, D>
{
  void (*operator*())(const unsigned *,
		      const unsigned *, const E *, const T*, T*)
    {
      return KERNELNAME<T_PASS E, N, S, W, V, D>;
    }
};
