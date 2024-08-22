
ArrayXXsMap
	buf_map      {buf_d     , 3, nCols},
	flux_out_map {flux_out_d, 3, nCols};
ArrayXXsCMap
	flux_in_map  {flux_in_d , 3, nCols},
	d_map        {d_d       , 1, nCols},
	v_map        {v_d       , 2, nCols},
	n_map        {n_d       , 1, nCols};

detail::friction_buf3(
	buf_map.col(i),
	flux_out_map.col(i),
	flux_in_map.col(i),
	d_map.col(i),
	v_map.col(i),
	n_map.col(i)
);
