#ifndef FORMAT_CONVERSION_H
#define FORMAT_CONVERSION_H

enum {MIN_SLICE_HEIGHT = 32, MAX_SLICE_HEIGHT = 512};

static int
estimate_slice_height(const int *row_ptr, int cur_row, int n_rows,
		      int n_threads, int n_elms)
{
  int slice_size;
  enum {MAX_WASTE_RATIO = 100, ELTS_PER_THREAD_RATIO = 4};
  float waste_per_row = 0.01 * n_elms * MAX_WASTE_RATIO / n_rows;
  int max_len = 0, waste = 0, used = 0;
  for (slice_size = 0;
       slice_size < n_threads && cur_row + slice_size < n_rows;
       slice_size++)
    {
      int len = (row_ptr[cur_row + slice_size + 1]
		 - row_ptr[cur_row + slice_size]);
      int nw, nu = used + len;
      if (len > max_len)
	{
	  nw = waste + slice_size * (len - max_len);
	}
      else
	{
	  nw = waste + max_len - len;
	  len = max_len;
	}
      if ((slice_size >= 32 && nw > waste_per_row * slice_size)
	  || (slice_size >= 1 && nu + nw > n_threads * ELTS_PER_THREAD_RATIO))
	break;
      max_len = len;
      waste = nw;
      used = nu;
    }
  /* Round down SLICE_SIZE to the power of two.  */
  slice_size |= slice_size >> 1;
  slice_size |= slice_size >> 2;
  slice_size |= slice_size >> 4;
  slice_size |= slice_size >> 8;
  slice_size |= slice_size >> 16;
  slice_size++;
  slice_size /= 2;

  return slice_size;
}

template<class T>
static void
copy_csr_elts_vslell(const csr_matrix<T> &A,
		     slell_matrix<T, host_memory_space_tag> &B,
		     int slice_size,
		     int cur_row)
{
  int max_len = 0;
  for (int j = 0; j < slice_size; j++)
    max_len = std::max(max_len,
		       A.row_ptr[cur_row + j + 1] - A.row_ptr[cur_row + j]);

  size_t old_size = B.cols.size();
  size_t new_size = B.cols.size() + max_len * slice_size;
  B.cols.resize(new_size);
  B.elms.resize(new_size);
  for (int j = cur_row; j < cur_row + slice_size; j++)
    {
      int k1 = 0, k;
      for (k = A.row_ptr[j]; k < A.row_ptr[j+1]; k++, k1++)
	{
	  B.cols[old_size + j - cur_row + k1 * slice_size] = A.cols[k];
	  B.elms[old_size + j - cur_row + k1 * slice_size] = A.elms[k];
	}
      for (; k1 < max_len; k1++)
	{
	  int last = (j > cur_row
		      ? B.cols[old_size + j - cur_row + k1 * slice_size - 1]
		      : 0);
	  B.cols[old_size + j - cur_row + k1 * slice_size] = last;
	}
    }
}

static int
hblk_maxslen(int *slen, int *blen,
	     int slice_size, int maxblen,
	     int hblock)
{
  int maxslen = 0;
  for (int i = 0; i < slice_size; i++)
    {
      int cb = std::min(maxblen, blen[i]);
      int cs = slen[i];
      cs -= maxblen - cb;
      cs += (blen[i] - cb) * hblock;
      maxslen = std::max(maxslen, cs);
    }
  return maxslen;
}

static int
hblk_cost(int *slen, int *blen,
	  int slice_size, int maxblen,
	  int hblock,
	  int smul, int bmul)
{
  int maxslen = hblk_maxslen(slen, blen, slice_size, maxblen, hblock);
  return smul * maxslen + bmul * maxblen;
}

template<class T>
static void
copy_csr_elts_hblocks(const csr_matrix<T> &A,
		      slell_matrix<T, host_memory_space_tag> &B,
		      int slice_size,
		      int cur_row,
		      int hblock)
{
  int *slen = new int[slice_size];
  int *blen = new int[slice_size];
  int minblen = INT_MAX, maxblen = 0;
  for (int rr = cur_row; rr < cur_row + slice_size; rr++)
    {
      int clen = 0, cblen = 0;
      int i;
      for (i = A.row_ptr[rr]; i + hblock - 1 < A.row_ptr[rr + 1];)
	if (!(A.cols[i] & (hblock - 1))
	    && A.cols[i + hblock - 1] - A.cols[i] == hblock - 1)
	  {
	    cblen++;
	    i += hblock;
	  }
	else
	  {
	    clen++;
	    i++;
	  }
      for (; i < A.row_ptr[rr + 1]; i++)
	clen++;
      slen[rr - cur_row]  = clen;
      blen[rr - cur_row] = cblen;
      maxblen = std::max(maxblen, cblen);
      minblen = std::min(minblen, cblen);
    }
  while (maxblen - minblen >= 3)
    {
      int d = (maxblen - minblen) / 3;
      int blen1 = minblen + d;
      int blen2 = blen1 + d;
      int cost1 = hblk_cost(slen, blen, slice_size, blen1, hblock,
				 sizeof(int) + sizeof(T),
				 sizeof(int) + hblock * sizeof(T));
      int cost2 = hblk_cost(slen, blen, slice_size, blen2,
				 sizeof(int) + sizeof(T), hblock,
				 sizeof(int) + hblock * sizeof(T));
      if (cost1 <= cost2)
	maxblen = blen2;
      else
	minblen = blen1;
    }
  int min_cost = INT_MAX, min_cost_blen = INT_MAX;
  for (; minblen <= maxblen; minblen++)
    {
      int cost = hblk_cost(slen, blen, slice_size, minblen, hblock,
				sizeof(int) + sizeof(T),
				sizeof(int) + hblock * sizeof(T));
      if (min_cost > cost)
	{
	  min_cost = cost;
	  min_cost_blen = minblen;
	}
    }
  maxblen = min_cost_blen;
  int maxslen = hblk_maxslen(slen, blen, slice_size, maxblen, hblock);
  int cofs = B.cols.size();
  int eofs = B.elms.size();
  int nse = maxslen * slice_size;
  int cbofs = cofs + nse;
  int ebofs = eofs + nse;
  int nbe = maxblen * slice_size;
  B.cols.resize(B.cols.size() + nse + nbe);
  B.elms.resize(B.elms.size() + nse + nbe * hblock);

  for (int rr = cur_row; rr < cur_row + slice_size; rr++)
    {
      int i, nsingle = 0, nblock = 0;
      for (i = A.row_ptr[rr]; i < A.row_ptr[rr + 1];)
	{
	  if (!(A.cols[i] & (hblock - 1))
	      && i + hblock - 1 < A.row_ptr[rr + 1]
	      && A.cols[i + hblock - 1] - A.cols[i] == hblock - 1
	      && nblock < maxblen)
	    {
	      B.cols[cbofs + rr - cur_row + nblock * slice_size] = A.cols[i] / hblock;
	      for (int j = 0; j < hblock; j++)
		B.elms[ebofs + rr - cur_row + (nblock * hblock + j) * slice_size] = A.elms[i + j];
	      nblock++;
	      i += hblock;
	    }
	  else if (nsingle < maxslen)
	    {
	      B.cols[cofs + rr - cur_row + nsingle * slice_size] = A.cols[i];
	      B.elms[eofs + rr - cur_row + nsingle * slice_size] = A.elms[i];
	      nsingle++;
	      i++;
	    }
	  else
	    {
	      B.cols[cbofs + rr - cur_row + nblock * slice_size] = A.cols[i] / hblock;
	      B.elms[ebofs + rr - cur_row + (nblock * hblock + A.cols[i] % hblock) * slice_size] = A.elms[i];
	      nblock++;
	      i++;
	    }
	}

      for (; nsingle < maxslen; nsingle++)
	{
	  int last = (rr > cur_row
		      ? B.cols[cofs + rr - cur_row + nsingle * slice_size - 1]
		      : 0);
	  B.cols[cofs + rr - cur_row + nsingle * slice_size] = last;
	}
      for (; nblock < maxblen; nblock++)
	{
	  int last = (rr > cur_row
		      ? B.cols[cbofs + rr - cur_row + nblock * slice_size - 1]
		      : 0);
	  B.cols[cbofs + rr - cur_row + nblock * slice_size] = last;
	}
    }
  delete[] slen;
  delete[] blen;
}

template<class T>
static int
find_diagonals(const csr_matrix<T> &A,
	       int slice_size,
	       int cur_row,
	       std::vector<int> &diag_pos)
{
  int rowlen[slice_size];
  int rowpos[slice_size];
  int maxlen = 0;
  int longrow = cur_row;
  for (int rr = cur_row; rr < cur_row + slice_size; rr++)
    {
      int len = A.row_ptr[rr + 1] - A.row_ptr[rr];
      rowlen[rr - cur_row] = len;
      rowpos[rr - cur_row] = A.row_ptr[rr];
      if (maxlen < len)
	{
	  maxlen = len;
	  longrow = rr;
	}
    }
  for (int i = A.row_ptr[longrow]; i < A.row_ptr[longrow + 1]; i++)
    {
      bool skip = false;
      for (int rr = cur_row; rr < cur_row + slice_size; rr++)
	{
	  while (rowpos[rr - cur_row] < A.row_ptr[rr + 1]
		 && A.cols[rowpos[rr - cur_row]] - rr < A.cols[i] - longrow)
	    rowpos[rr - cur_row]++;
	  if (!(rowpos[rr - cur_row] < A.row_ptr[rr + 1]
		&& A.cols[rowpos[rr - cur_row]] - rr == A.cols[i] - longrow)
	      && rowlen[rr - cur_row] == maxlen)
	    {
	      skip = true;
	      break;
	    }
	}
      if (skip)
	continue;
      diag_pos.push_back(A.cols[i] - (longrow - cur_row) + slice_size);
      maxlen--;
      for (int rr = cur_row; rr < cur_row + slice_size; rr++)
	{
	  if (rowpos[rr - cur_row] < A.row_ptr[rr + 1]
	      && A.cols[rowpos[rr - cur_row]] - rr == A.cols[i] - longrow)
	    rowlen[rr - cur_row]--;
	}
    }
  return maxlen;
}

template<class T>
static void
copy_csr_elts_diags(const csr_matrix<T> &A,
		    slell_matrix<T, host_memory_space_tag> &B,
		    int slice_size,
		    int cur_row,
		    std::vector<int> &diag_pos)
{
  int diag_start = diag_pos.size();
  int maxlen = find_diagonals(A, slice_size, cur_row, diag_pos);

  int ccofs = B.cols.size();
  int ecofs = B.elms.size();
  int nce = maxlen * slice_size;
  int edofs = ecofs + nce;
  int nde = (diag_pos.size() - diag_start) * slice_size;
  B.cols.resize(B.cols.size() + nce);
  B.elms.resize(B.elms.size() + nce + nde);
  for (int rr = cur_row; rr < cur_row + slice_size; rr++)
    {
      int di = diag_start;
      int thisrow_nc = 0;
      for (int i = A.row_ptr[rr]; i < A.row_ptr[rr + 1]; i++)
	{
	  int c = A.cols[i], d = c - (rr - cur_row) + slice_size;
	  int n_diags = diag_pos.size();
	  while (di < n_diags
		 && diag_pos[di] < d)
	    di++;
	  if (di < n_diags
	      && diag_pos[di] == d)
	    {
	      int nd = di - diag_start;
	      B.elms[edofs + rr - cur_row + nd * slice_size] = A.elms[i];
	    }
	  else
	    {
	      B.cols[ccofs + rr - cur_row + thisrow_nc * slice_size] = c;
	      B.elms[ecofs + rr - cur_row + thisrow_nc * slice_size] = A.elms[i];
	      thisrow_nc++;
	    }
	}
      for (; thisrow_nc < maxlen; thisrow_nc++)
	{
	  int last = (rr > cur_row
		      ? B.cols[ccofs + rr - cur_row + thisrow_nc * slice_size - 1]
		      : 0);
	  B.cols[ccofs + rr - cur_row + thisrow_nc * slice_size] = last;
	}
    }
}

template<class T>
slell_matrix<T, host_memory_space_tag>
csr_to_slell(const csr_matrix<T> &A, int req_slice_height, int hblock,
	     bool var_height, bool use_diags, int min_height = 4)
{
  const_cast<csr_matrix<T> &>(A).pad(MAX_SLICE_HEIGHT);
  slell_matrix<T, host_memory_space_tag> B(A);
  if (hblock > 1)
    use_diags = false;
  B.slice_height = req_slice_height;
  B.hblock = hblock;
  B.var_height = var_height;
  B.diags = use_diags;
  B.n_slices = 0;
  B.cols.reserve(A.cols.size() * 3 / 2);
  B.elms.reserve(A.cols.size() * 3 / 2);
  int slice_size;
  int slice_ptr_inc = 1;
  B.slice_ptr.push_back(0);
  if (hblock > 1)
    {
      B.slice_ptr.push_back(0);
      slice_ptr_inc++;
    }
  if (var_height)
    {
      B.slice_ptr.push_back(0);
      slice_ptr_inc++;
    }

  if (use_diags)
    {
      B.slice_ptr.push_back(0);
      B.slice_ptr.push_back(0);
      slice_ptr_inc += 2;
    }
  std::vector<int> diag_pos;
  for (int i = 0; i < A.n_rows; i += slice_size)
    {
      if (var_height)
	slice_size = estimate_slice_height(&A.row_ptr[0], i, A.n_rows,
					   req_slice_height, A.elms.size());
      else
	slice_size = req_slice_height;

      if (slice_size < min_height)
	slice_size = min_height;

      if (use_diags)
	copy_csr_elts_diags(A, B, slice_size, i, diag_pos);
      else if (hblock > 1)
	copy_csr_elts_hblocks(A, B, slice_size, i, hblock);
      else
	copy_csr_elts_vslell(A, B, slice_size, i);

      B.slice_ptr.push_back(B.elms.size());
      if (var_height)
	B.slice_ptr.push_back(i + slice_size);
      if (hblock > 1)
	B.slice_ptr.push_back(B.cols.size());
      if (use_diags)
	{
	  B.slice_ptr.push_back(B.cols.size());
	  B.slice_ptr.push_back(diag_pos.size());
	}
      B.n_slices++;
    }
  if (use_diags)
    {
      int dp_off = var_height ? 3 : 2;
      for (unsigned i = 0; i <= B.n_slices; i++)
	B.slice_ptr[dp_off + i * slice_ptr_inc] += B.slice_ptr.size();
      B.slice_ptr.reserve(B.slice_ptr.size() + diag_pos.size());
      B.slice_ptr.insert(B.slice_ptr.end(), diag_pos.begin(), diag_pos.end());
    }
  return B;
}

template<class T>
slell_matrix<T, host_memory_space_tag>
csr_to_slell_guess_format(const csr_matrix<T> &A, int examine_ratio = 1, int min_height = 4)
{
  enum {MAX_WASTE_RATIO = 100, ELTS_PER_THREAD = 8, N_THREADS = 256,
	MIN_HBLOCK_RATIO = 25,
	MIN_DIAGS_RATIO = 50,
        ELTS_PER_SLICE = ELTS_PER_THREAD * N_THREADS};
  int rows_to_examine = (A.n_rows * examine_ratio + 99) / 100;
  int trial_slice_sz = 32;

  rows_to_examine = (rows_to_examine + trial_slice_sz - 1) & ~(trial_slice_sz - 1);

  int elts_to_examine = std::max(1, A.row_ptr[rows_to_examine + 1]);

  int waste = 0;
  int n_hblocks2 = 0;
  int n_diags = 0;
  for (int i = 0; i < rows_to_examine; i += trial_slice_sz)
    {
      int maxlen = 0;
      for (int r = i; r < i + trial_slice_sz; r++)
	maxlen = std::max(maxlen, A.row_ptr[r + 1] - A.row_ptr[r]);
      for (int r = i; r < i + trial_slice_sz; r++)
	{
	  waste += maxlen - (A.row_ptr[r + 1] - A.row_ptr[r]);
	  int hblock = 2;
	  for (int i = A.row_ptr[r]; i + hblock - 1 < A.row_ptr[r + 1]; i++)
	    if (!(A.cols[i] % hblock)
		&& A.cols[i + hblock - 1] == A.cols[i] + hblock - 1)
	      {
		n_hblocks2++;
		i += hblock - 1;
	      }
	}
      {
	std::vector<int> dummy;
	find_diagonals(A, trial_slice_sz, i, dummy);
	n_diags += dummy.size();
      }
    }

  int waste_ratio = 100 * waste / elts_to_examine;

  bool want_var_height = (waste_ratio > MAX_WASTE_RATIO);

  float avg_elts_per_row = 1.f * elts_to_examine / rows_to_examine;
  int want_slice_height;

  if (want_var_height)
    want_slice_height = N_THREADS;
  else
    {
      want_slice_height = ELTS_PER_SLICE / avg_elts_per_row;

      want_slice_height |= want_slice_height >> 1;
      want_slice_height |= want_slice_height >> 2;
      want_slice_height |= want_slice_height >> 4;
      want_slice_height |= want_slice_height >> 8;
      want_slice_height |= want_slice_height >> 16;
      want_slice_height = (want_slice_height + 1) >> 1;

      if (want_slice_height < MIN_SLICE_HEIGHT)
	want_slice_height = MIN_SLICE_HEIGHT;
      else if (want_slice_height > MAX_SLICE_HEIGHT)
	want_slice_height = MAX_SLICE_HEIGHT;
    }

  int want_hblocks = 1;
  bool want_diags = (n_diags * trial_slice_sz
		     > elts_to_examine * MIN_DIAGS_RATIO / 100);
  if (!want_diags && n_hblocks2 > elts_to_examine * MIN_HBLOCK_RATIO / 100)
    want_hblocks = 2;

  return csr_to_slell(A, want_slice_height, want_hblocks, want_var_height,
		      want_diags, min_height);
}

#endif
