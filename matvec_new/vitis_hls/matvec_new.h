#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include "assert.h"

constexpr unsigned int UNROLL_FACTOR = 4;
typedef ap_int<32> dtype;


const unsigned int MAX_LEN = 1024 * 16;
