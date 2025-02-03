#include "matvec_new.h"


//inline void load_array(
//	dtype *global,
//	dtype *local,
//	unsigned int goffset,
//	unsigned int loffset,
//	unsigned int num_words
//){
//  l_load: for(unsigned int i=0; i<num_words; i++){
//    #pragma HLS loop_tripcount min=4 max=(1024 * 16) avg=1024
//    #pragma HLS dependence variable=global type=inter false
//    #pragma HLS dependence variable=global type=intra false
//    #pragma HLS dependence variable=local type=inter false
//    #pragma HLS dependence variable=local type=intra false
//    local[loffset + i] = global[goffset + i];
//  }
//}

inline void load_array(
	dtype *global,
	dtype *local,
	unsigned int goffset,
	unsigned int loffset,
	unsigned int num_words
){
#pragma HLS dependence variable=global type=inter false
#pragma HLS dependence variable=global type=intra false
#pragma HLS dependence variable=local type=inter false
#pragma HLS dependence variable=local type=intra false
	l_load: for(unsigned int i=0; i<num_words; i+=UNROLL_FACTOR){
		#pragma HLS loop_tripcount min=1 max=4096 avg=256
		#pragma HLS PIPELINE II=1
		for(unsigned int j=0; j<UNROLL_FACTOR; j++){
			#pragma HLS UNROLL
			local[loffset + i + j] = global[goffset + i + j];
		}
	}
}

//inline void store_array(
//	dtype *global,
//	dtype *local,
//	unsigned int goffset,
//	unsigned int loffset,
//    unsigned int num_words
//){
//  l_store: for(unsigned int i=0; i<num_words; i++){
//    #pragma HLS loop_tripcount min=4 max=(1024 * 16) avg=1024
//    #pragma HLS dependence variable=global type=inter false
//    #pragma HLS dependence variable=global type=intra false
//    #pragma HLS dependence variable=local type=inter false
//    #pragma HLS dependence variable=local type=intra false
//    global[goffset + i] = local[loffset + i];
//  }
//}

inline void store_array(
	dtype *global,
	dtype *local,
	unsigned int goffset,
	unsigned int loffset,
	unsigned int num_words
){
#pragma HLS dependence variable=global type=inter false
#pragma HLS dependence variable=global type=intra false
#pragma HLS dependence variable=local type=inter false
#pragma HLS dependence variable=local type=intra false
	l_store: for(unsigned int i=0; i<num_words; i+=UNROLL_FACTOR){
		#pragma HLS loop_tripcount min=1 max=4096 avg=256
		#pragma HLS PIPELINE II=1
		for(unsigned int j=0; j<UNROLL_FACTOR; j++){
			#pragma HLS UNROLL
			global[goffset + i + j] = local[loffset + i + j];
		}
	}
}

// top-level function
void matvec_new(
	dtype *mat,
	dtype *vec,
	dtype *out,
	unsigned int vec_len,
	unsigned int out_len
) {
#pragma HLS INTERFACE mode=s_axilite port=return bundle=control
#pragma HLS INTERFACE mode=s_axilite port=vec_len bundle=control
#pragma HLS INTERFACE mode=s_axilite port=out_len bundle=control
#pragma HLS INTERFACE mode=m_axi port=mat depth=16384 offset=slave bundle=gmem_out0
#pragma HLS INTERFACE mode=s_axilite port=mat bundle=control
#pragma HLS INTERFACE mode=m_axi port=vec depth=16384 offset=slave bundle=gmem_in0
#pragma HLS INTERFACE mode=s_axilite port=vec bundle=control
#pragma HLS INTERFACE mode=m_axi port=out depth=16384 offset=slave bundle=gmem_out0
#pragma HLS INTERFACE mode=s_axilite port=out bundle=control

#pragma HLS dependence variable=mat type=inter false
#pragma HLS dependence variable=mat type=intra false
#pragma HLS dependence variable=vec type=inter false
#pragma HLS dependence variable=vec type=intra false
#pragma HLS dependence variable=out type=inter false
#pragma HLS dependence variable=out type=intra false

//  dtype mat_buf [MAX_LEN];
  dtype vec_buf [MAX_LEN];
  dtype out_buf [MAX_LEN];
//#pragma HLS ARRAY_PARTITION variable=mat_buf type=cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=vec_buf type=cyclic factor=4
//#pragma HLS ARRAY_PARTITION variable=out_buf type=cyclic factor=4
//#pragma HLS dependence variable=mat_buf type=inter false
//#pragma HLS dependence variable=mat_buf type=intra false
#pragma HLS dependence variable=vec_buf type=inter false
#pragma HLS dependence variable=vec_buf type=intra false

  assert(out_len % 16 == 0);
  assert(vec_len % 16 == 0);
  assert(out_len > 0);
  assert(vec_len > 0);
  const unsigned int num_comp = (out_len >= MAX_LEN) ? MAX_LEN : out_len;
  const unsigned int num_words = (vec_len >= MAX_LEN) ? MAX_LEN : vec_len;

  load_array(vec, vec_buf, 0, 0, num_words);

  //  l_comp: for(unsigned int step=0; step<num_comp; step++){
//    #pragma HLS loop_tripcount min=4 max=(1024 * 16) avg=1024
//    load_array(mat, mat_buf, step * num_words, 0, num_words);
//    dtype sum = 0;
//    l_innerproduct: for(unsigned int i=0; i<num_words; i++){
//      #pragma HLS loop_tripcount min=4 max=(1024 * 16) avg=1024
//      #pragma HLS PIPELINE II=1
//      #pragma HLS UNROLL factor=4
//      sum += mat_buf[i] * vec_buf[i];
//    }
//    out_buf[step] = sum;
//  }

  unsigned int in_addr = 0, out_addr = 0;
  dtype sum = 0;
  for(unsigned int addr = 0; addr < num_comp * num_words; addr+=UNROLL_FACTOR){
	  #pragma HLS loop_tripcount min=4 max=67108864 avg=262144
	  #pragma HLS PIPELINE II=1
	  for(unsigned int i = 0; i < UNROLL_FACTOR; i++){
			#pragma HLS UNROLL
		  sum += mat[addr + i] * vec_buf[in_addr + i];
	  }
	  in_addr += UNROLL_FACTOR;
	  if (in_addr == num_words){
		  out_buf[out_addr++] = sum;
		  sum = 0;
		  in_addr = 0;
	  }
  }

  store_array(out, out_buf, 0, 0, num_comp);
}
