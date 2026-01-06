#include <fix16.h>
#include <mlp.h>
#include <neslib.h>
#include <stdlib.h>
#include <string.h>
#include <weights.h>

#define WHITE_COLOR_PALETTE 0x30

void put_str(unsigned int adr, const char *str)
{
    vram_adr(adr);
    // Iterate until we find a \0 character (end of string)
    while(*str)
    {
        // -0x20 because ASCII code 0x20 is placed in tile 0 of the CHR
        vram_put((*str++)-0x20);
    }
}

/*
Draw your input here (0-10)
{0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0}
*/
static fix16_t _data_input[MLP_INPUT_SIDE][MLP_INPUT_SIDE] = {
    {0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 10, 0, 0, 0},
    {0, 0, 10, 10, 0, 0, 0},
    {0, 0, 0, 10, 0, 0, 0},
    {0, 0, 0, 10, 0, 0, 0},
    {0, 0, 0, 10, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0},
};

static fix16_t _mlp_output[MLP_OUTPUT_SIZE];

void main() {
    uint8_t i, j;
    int intValue;

    fix16_t fix16_ten = fix16_from_int(10);
    fix16_t fix16_one_hundred = fix16_from_int(100);

    // Normalize input array from [0, 10] range to [0, 1]
    for (i = 0; i < MLP_INPUT_SIDE; i++) {
        for (j = 0; j < MLP_INPUT_SIDE; j++) {
            intValue = (int) _data_input[i][j];
            _data_input[i][j] = fix16_div(fix16_from_int(intValue), fix16_ten);
        }
    }

    // Rendering is disabled at the startup, the palette is all black
    pal_col(1, WHITE_COLOR_PALETTE);

    // You can't put data into vram through vram_put while rendering is enabled
    // so you have to disable rendering to put things like text or a level map
    // into the nametable
    // However, there is a way to update small number of nametable tiles while rendering
    // is enabled, using set_vram_update and an update list

    // The screen has 32 columns and 30 rows sprites, with the first sprite at the top left being NTADR_A(0,1)
    put_str(NTADR_A(1,2), "NES AI");

    // Processing input using the neural network
    mlp_forward((const fix16_t*) _data_input, _mlp_output);

    // Print % probability for each output
    for (i = 0; i < MLP_OUTPUT_SIZE; i++) {
        char tempBuffer[4];
        char buffer[12];

        // 0: 85%
        // 1: 4%
        // ...
        itoa(i, tempBuffer, 10);
        strcpy(buffer, tempBuffer);
        strcat(buffer, ": ");

        intValue = fix16_to_int(fix16_mul(_mlp_output[i], fix16_one_hundred));
        itoa(intValue, tempBuffer, 10);
        strcat(buffer, tempBuffer);
        strcat(buffer, "%");

        put_str(NTADR_A(1,i+4), buffer);
    }

    // Enable rendering
    ppu_on_all();

    // Do nothing, infinite loop
    while(1);
}