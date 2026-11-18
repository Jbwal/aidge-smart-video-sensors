/*
 * SPDX-License-Identifier: LicenseRef-Asygn-BSD3-NoReverse-NoExternalUse
 *
 * Copyright (c) 2026 Asygn SAS
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form, except as embedded into a Asygn SAS
 *    integrated circuit in a product or a software update for such product, must
 *    reproduce the above copyright notice, this list of conditions and the
 *    following disclaimer in the documentation and/or other materials provided
 *    with the distribution.
 *
 * 3. Neither the name of Asygn SAS nor the names of its contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * 4. This software, with or without modification, must only be used with a Asygn
 *    SAS integrated circuit.
 *
 * 5. Any software provided in binary form under this license must not be reverse
 *    engineered, decompiled, modified and/or disassembled.
 *
 * THIS SOFTWARE IS PROVIDED BY ASYGN SAS "AS IS" AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL ASYGN SAS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#define NNPA_CTX_IMPLEMENTATION

#include <stdio.h>
#include <stdint.h>

#include "dnn/include/kernels/nnpa.hpp"
#include "dnn/include/layers/_0__NNPA_0.h"
#include "data/_0__NNPA_0_input_0.h"

#define PRINT_EVERY_N_INFERENCES 100


static void convert_i8_to_u8(const int8_t *src, uint8_t *dst, uint32_t size)
{
    for (uint32_t i = 0; i < size; i++) {
        dst[i] = (uint8_t)((int16_t)src[i] + 128);
    }
}


static void print_outputs_u8(const uint8_t *output, uint32_t size)
{
    printf("Outputs:");

    for (uint32_t i = 0; i < size; i++) {
        printf(" %u", (unsigned int)output[i]);
    }

    printf("\n");
}


static void print_outputs_volatile_u8(const volatile uint8_t *output, uint32_t size)
{
    printf("Outputs:");

    for (uint32_t i = 0; i < size; i++) {
        printf(" %u", (unsigned int)output[i]);
    }

    printf("\n");
}


int main()
{
    uint32_t inference_counter = 0;

    const uint32_t input_size0 =
        sizeof(_0__NNPA_0_input_0) /
        sizeof(_0__NNPA_0_input_0[0]);

    const uint32_t output_size0 =
        _0__NNPA_0_OUTPUT_0_NB_OUTPUTS *
        _0__NNPA_0_OUTPUT_0_OUT_HEIGHT *
        _0__NNPA_0_OUTPUT_0_OUT_WIDTH *
        _0__NNPA_0_OUTPUT_0_OUT_BATCH;

    static uint8_t __NNPA_0_input_0_u8[
        sizeof(_0__NNPA_0_input_0) /
        sizeof(_0__NNPA_0_input_0[0])
    ];

    _0__NNPA_0.configure(input_size0, output_size0);

    _0__NNPA_0.init();

    printf("NNPA initialized\n");
    printf_flush();

    convert_i8_to_u8(_0__NNPA_0_input_0, __NNPA_0_input_0_u8, input_size0);

    _0__NNPA_0.load_input(__NNPA_0_input_0_u8);

    printf("Input Loaded \n");
    printf_flush();

    while (1) {
        USER_LED_OFF();
        _0__NNPA_0.run();

        volatile uint8_t *_0__NNPA_0_output_0 =
            _0__NNPA_0.get_output_ptr();

        inference_counter++;

#if PRINT_EVERY_N_INFERENCES > 0
        if ((inference_counter % PRINT_EVERY_N_INFERENCES) == 0) {
            printf("%lu inferences run\n", (unsigned long)inference_counter);
            {
                const uint32_t output_size0 = 96000;
                const uint32_t output_print_size0 =
                    output_size0 <= 10 ? output_size0 : 10;

                printf(
                    "Large output detected (%lu values). "
                    "Printing first %lu values only:\n",
                    (unsigned long)output_size0,
                    (unsigned long)output_print_size0
                );

                print_outputs_volatile_u8(
                    _0__NNPA_0_output_0,
                    output_print_size0
                );
            }
            printf_flush();
        }
#endif

        USER_LED_ON();
        colibrySCU->delay_us(1);
    }

    return 0;
}