#ifndef KERNELS_NNPA_HPP
#define KERNELS_NNPA_HPP

#include <stdio.h>
#include "chip.h"
#include "colibry_drivers.h"
#include "colibry_board.hpp"


#ifndef CHIP_VDD_VOLTAGE
#define CHIP_VDD_VOLTAGE            (0.76 + 0.0125)
#endif

#ifndef HFCLK_FREQUENCY_MHZ
#define HFCLK_FREQUENCY_MHZ         40
#endif

#ifndef APB_CLK_DIV
#define APB_CLK_DIV                 4
#endif

#ifndef BITSTREAM_FLASH_ADDR
#define BITSTREAM_FLASH_ADDR        0x20000
#endif

#ifndef SPIS_DMA_CHANNEL
#define SPIS_DMA_CHANNEL            3
#endif

#ifndef APP_SPI_RF_BRIDGE_OPCODE
#define APP_SPI_RF_BRIDGE_OPCODE    0x9B
#endif

#ifndef APP_SPI_LFCLK_DIS_OPCODE
#define APP_SPI_LFCLK_DIS_OPCODE    0x20
#endif

#ifndef APP_RF_DEMO_ID
#define APP_RF_DEMO_ID              0x54AC
#endif


extern const int __sram_user_start;
extern const int __sram_buffer_size;

#ifdef NNPA_CTX_IMPLEMENTATION

volatile uint8_t nnpa_inference_done = 0;
ColibryBoard colibryBoard;


irq_handler_t nnpa_irq_handler(void) {
    if (NNPA_reg->nnpa_int_status.fields.evt_exec_end) {
        nnpa_inference_done = 1;
        NNPA_reg->nnpa_int_clear.word = EVT_EXEC_END_MASK;
    }
}
#else

extern volatile uint8_t nnpa_inference_done;
extern ColibryBoard colibryBoard;

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////
// NNPA context
/////////////////////////////////////////////////////////////////////////////////////////////////////

class nnpa_ctx {
public:
    nnpa_ctx(size_t input_size, size_t output_size)
        : input_size_(input_size),
          output_size_(output_size),
          initialized_(0)
    {
    }

    ~nnpa_ctx() {}

    void configure(size_t input_size, size_t output_size)
    {
        input_size_ = input_size;
        output_size_ = output_size;
    }

    void init()
    {
        if (initialized_) {
            return;
        }

        COLIBRY_DRIVERS_INIT();

        board_init_cfg_t board_init_cfg;
        uint32_t nb_sram_buffers_used;

        const int sram_buffers_size = (int)&__sram_buffer_size;

        colibrySCU->regmap->scu_periph_clk_en.word = 0xFFFFFFFF;

        board_init_cfg.vdd_setting = CHIP_VDD_VOLTAGE;
        board_init_cfg.hfclk_freq_mhz = HFCLK_FREQUENCY_MHZ;
        board_init_cfg.apb_clk_fdiv = APB_CLK_DIV;
        board_init_cfg.nnpa_clk_fdiv = 1;

        board_init_cfg.spis_enable = 0;
        board_init_cfg.spim_rf_enable = 1;
        board_init_cfg.spim_acc_enable = 0;

        board_init_cfg.spis_dma_channel = SPIS_DMA_CHANNEL;
        board_init_cfg.spim0_csn_margin = 100;

        colibryBoard.global_init(board_init_cfg);

        printf("---------------------------\n");
        printf("NNPA ctx init from main\n");
        printf("---------------------------\n");
        printf_flush();

        // Fixed SPI packet header to nRF
        colibryBoard.spim0_tx_buff[0] = APP_SPI_RF_BRIDGE_OPCODE;
        colibryBoard.spim0_tx_buff[1] = APP_RF_DEMO_ID & 0xFF;
        colibryBoard.spim0_tx_buff[2] = (APP_RF_DEMO_ID >> 8) & 0xFF;

        // Disable LFCLK on nRF
        colibryBoard.spim0_tx_buff[0] = APP_SPI_LFCLK_DIS_OPCODE;
        colibryBoard.spim0_exchange(1);
        colibryBoard.spim0_tx_buff[0] = APP_SPI_RF_BRIDGE_OPCODE;

        printf("Loading bitstream...\n");
        printf_flush();

        colibryBoard.copy_bitstream_to_mem(
            BITSTREAM_FLASH_ADDR,
            NNPAMEM_BASE_ADDR
        );

        if (colibryBoard.check_nnpa_bitstream()) {
            printf("ERROR: NNPA bitstream check failed\n");
            printf_flush();
            exit(1);
        }

        printf("Bitstream OK\n");
        printf_flush();

        colibryBoard.nnpa_allocate_buffers();

        USER_LED_ON();
        colibrySCU->delay_ms(100);
        USER_LED_OFF();

        // Enable NNPA end interrupt
        NNPA_reg->nnpa_int_clear.word = EVT_EXEC_END_MASK;
        NNPA_reg->nnpa_int_en.word = EVT_EXEC_END_MASK;

        // Keep UART for debug prints.
        // For power measurement, remove SCU_UART_CLK_EN_MASK.
        colibrySCU->regmap->scu_periph_clk_en.word =
            SCU_APB_CLK_EN_MASK  |
            SCU_NNPA_CLK_EN_MASK |
            SCU_PORT_CLK_EN_MASK |
            SCU_UART_CLK_EN_MASK;

        nb_sram_buffers_used =
            ((uint32_t)&__sram_user_start - 0x10000000) / sram_buffers_size + 1;

        colibrySCU->regmap->sram_cuts_pd_ctrl.word =
            (uint32_t)~((1 << nb_sram_buffers_used) - 1);

        colibryBoard.extflash_power_disable();
        colibryBoard.sensors_power_disable();
        colibryBoard.configure_lowpower_io();

        initialized_ = 1;

        printf("NNPA ctx initialized\n");
        printf_flush();
    }

    template <class Input_T>
    void load_input(const Input_T *input)
    {
        if (!initialized_) {
            init();
        }

        const uint8_t *input_u8 = (const uint8_t *)input;

        const uint32_t input_size_bytes =
            (uint32_t)(input_size_ * sizeof(Input_T));

        printf("input_size_=%lu sizeof(Input_T)=%lu\n",
            (unsigned long)input_size_,
            (unsigned long)sizeof(Input_T));

        printf("Loading input: %lu bytes\n", (unsigned long)input_size_bytes);
        printf_flush();

        NNPA_reg->nnpa_control.fields.cpu_mem_access = 1;

        NNPA_reg->input_loader_cfg.word =
            (3 << LOADER_NBYTE_VALID_OFFSET) |
            (0 << LOADER_NBIT_MODE_OFFSET)   |
            (0 << LOADER_SIGNED_DATA_OFFSET);

        NNPA_reg->nnpa_trig_control.word = INPUT_LOADER_RST_MASK;

        const uint32_t n_full_words = input_size_bytes / 4;

        for (uint32_t i = 0; i < n_full_words; i++) {
            uint32_t input_data_word =
                *(const uint32_t *)(input_u8 + i * 4);

            NNPA_reg->input_loader_data.word = input_data_word;
        }

        printf("Input loaded\n");
        printf_flush();
    }
    
    template <class Output_T>
    void run(Output_T *output)
    {
        if (!initialized_) {
            init();
        }

        NNPA_reg->nnpa_control.fields.cpu_mem_access = 0;
        NNPA_reg->nnpa_trig_control.word = RUN_START_MASK;

        while (nnpa_inference_done == 0) {
            colibryBoard.enter_low_power(LOW_POWER_INFERENCE_MODE);
        }

        nnpa_inference_done = 0;
        NNPA_reg->nnpa_control.fields.cpu_mem_access = 1;

        read_output(output);
    }
    void run()
    {
        if (!initialized_) {
            init();
        }

        NNPA_reg->nnpa_control.fields.cpu_mem_access = 0;
        NNPA_reg->nnpa_trig_control.word = RUN_START_MASK;

        while (nnpa_inference_done == 0) {
            colibryBoard.enter_low_power(LOW_POWER_INFERENCE_MODE);
        }

        nnpa_inference_done = 0;
        NNPA_reg->nnpa_control.fields.cpu_mem_access = 1;
    }

    volatile uint8_t *get_output_ptr()
    {
        uint32_t nn_output_addr =
            NNPA_NBYTE_PER_MEM_WORD *
            NNPA_reg->ofmap_start_maddr.word +
            NNPAMEM_BASE_ADDR;

        return (volatile uint8_t *)nn_output_addr;
    }

    template <class Input_T, class Output_T>
    void run(const Input_T *input, Output_T *output)
    {
        load_input(input);
        run(output);
    }

private:
    template <class Output_T>
    void read_output(Output_T *output)
    {
        const uint32_t output_size_bytes =
            (uint32_t)(output_size_ * sizeof(Output_T));

        uint32_t nn_output_addr =
            NNPA_NBYTE_PER_MEM_WORD * NNPA_reg->ofmap_start_maddr.word +
            NNPAMEM_BASE_ADDR;

        volatile uint8_t *src = (volatile uint8_t *)nn_output_addr;
        uint8_t *dst = (uint8_t *)output;

        for (uint32_t i = 0; i < output_size_bytes; i++) {
            dst[i] = src[i];
        }
    }

private:
    size_t input_size_;
    size_t output_size_;
    uint8_t initialized_;
};

#endif