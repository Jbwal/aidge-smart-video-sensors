
#include <stdint.h>
#include <stddef.h>


#include "forward.hpp"

// Layer & memory configurations

#include "layers/_0__NNPA_0.h"
#include "kernels/nnpa.hpp"

// Memory block
static unsigned char mem_0[384000];


void model_forward (const int8_t* _0__NNPA_0_input_0,float** _0__NNPA_0_output_0_ptr)
{
    
    // == _0__NNPA_0 ===========================================================
    float* _0__NNPA_0_output_0 = (float*) (mem + _0__NNPA_0_OUTPUT_0_MEM_OFFSET);
    _0__NNPA_0.run(_0__NNPA_0_input_0, _0__NNPA_0_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _0__NNPA_0\n");
    FILE* _0__NNPA_0_OUTPUT_0_STREAM = fopen("data/export_outputs/_0__NNPA_0_output_0.txt", "w");
    export_cpp::saveOutputs<_0__NNPA_0_OUTPUT_0_NB_OUTPUTS,
                            _0__NNPA_0_OUTPUT_0_OUT_HEIGHT,
                            _0__NNPA_0_OUTPUT_0_OUT_WIDTH,
                            _0__NNPA_0_OUTPUT_0_MEM_CONT_OFFSET,
                            _0__NNPA_0_OUTPUT_0_MEM_CONT_SIZE,
                            _0__NNPA_0_OUTPUT_0_MEM_WRAP_OFFSET,
                            _0__NNPA_0_OUTPUT_0_MEM_WRAP_SIZE,
                            _0__NNPA_0_OUTPUT_0_DEV_FMT>
                            (_0__NNPA_0_output_0,
                            _0__NNPA_0_OUTPUT_0_STREAM);
    fclose(_0__NNPA_0_OUTPUT_0_STREAM);
    #endif

    
    *_0__NNPA_0_output_0_ptr = _0__NNPA_0_output_0;
}
