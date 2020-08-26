#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include <xtime_l.h>
#include "parameter.h"
#include "ifmap_fmap_integer.h"
#include "cnn_func_integer.h"
#include "ifmap_fmap_float.h"
#include "cnn_func_float.h"
//#include "benchmarking.h"
//#include "benchmarking.c"
#include "address_map_arm.h"

#define KEY_BASE              0xFF200050
#define VIDEO_IN_BASE         0xFF203060
#define FPGA_ONCHIP_BASE      0xC8000000
#define LED_BASE 							0xFF200000

volatile int * KEY_ptr				= (int *) KEY_BASE;
volatile int * Video_In_DMA_ptr	= (int *) VIDEO_IN_BASE;
volatile short * Video_Mem_ptr	= (short *) FPGA_ONCHIP_BASE;
volatile int * LED_ptr = (int *)LED_BASE; // red LED address


#define TEST_CASE 2
#define TEST_ROUNDS 10

float ofmap_ref[_data][_class] = { { 0, }, };
short ofmap_opt[_data][_class] = { { 0, }, };

unsigned int initializor_dummy(unsigned int uiParam0, unsigned int uiParam1, unsigned int uiParam2, unsigned int uiParam3){
	return 1;
}

unsigned int validator_dummy(unsigned int uiParam0, unsigned int uiParam1, unsigned int uiParam2, unsigned int uiParam3 ){
	return 1;
}

void cnn_ref(float *ofmap, float *ifmap, int data_set){
	int i = 0;

	float *ofmap1 = (float *) calloc(E_C1*F_C1*M_C1*N_C1, sizeof(float));
	float *ofmap2 = (float *) calloc(E_P1*F_P1*C_P1*N_P1, sizeof(float));
	float *ofmap3 = (float *) calloc(E_C2*F_C2*M_C2*N_C2, sizeof(float));
	float *ofmap4 = (float *) calloc(E_P2*F_P2*C_P2*N_P2, sizeof(float));
	float *ofmap5 = (float *) calloc(E_C3*F_C3*M_C3*N_C3, sizeof(float));
	float *ofmap6 = (float *) calloc(E_R1*F_R1*M_R1*N_R1, sizeof(float));

	//Input mapping
	for (i = 0; i<H_C1*W_C1; i++) {
		ifmap[i] = data[data_set][i];
	}

	/////////////////////////////
	//         Layer #1       //
	////////////////////////////
	convolution_f(ofmap1, ifmap, fmap1_f, N_C1, C_C1, M_C1, F_C1, E_C1, R_C1, S_C1, H_C1, W_C1, U_C1);
	bias_f(ofmap1, ofmap1, bias1_f, N_C1, M_C1, E_C1, F_C1);
	pool_f(ofmap2, ofmap1, E_C1, F_C1, M_C1);

	/////////////////////////////
	//         Layer #2       //
	////////////////////////////
	convolution_f(ofmap3, ofmap2, fmap2_f, N_C2, C_C2, M_C2, F_C2, E_C2, R_C2, S_C2, H_C2, W_C2, U_C2);
	bias_f(ofmap3, ofmap3, bias2_f, N_C2, M_C2, E_C2, F_C2);
	pool_f(ofmap4, ofmap3, E_C2, F_C2, M_C2);

	/////////////////////////////
	//         Layer #3       //
	////////////////////////////
	convolution_f(ofmap5, ofmap4, fmap3_f, N_C3, C_C3, M_C3, F_C3, E_C3, R_C3, S_C3, H_C3, W_C3, U_C3);
	bias_f(ofmap5, ofmap5, bias3_f, N_C3, M_C3, E_C3, F_C3);
	relu_f(ofmap6, ofmap5, E_C3, F_C3, M_C3);

	/////////////////////////////
	//         Layer #4       //
	////////////////////////////
	convolution_f(ofmap, ofmap6, fmap4_f, N_C4, C_C4, M_C4, F_C4, E_C4, R_C4, S_C4, H_C4, W_C4, U_C4);
	bias_f(ofmap, ofmap, bias4_f, N_C4, M_C4, E_C4, F_C4);

	free(ofmap1);
	free(ofmap2);
	free(ofmap3);
	free(ofmap4);
	free(ofmap5);
	free(ofmap6);
}

void cnn_opt(short *ofmap, short *ifmap, int data_set){
	///////////////////////////////////////////////////////////////
	// You are allowed to modify only the inside of this function
	// From this line

	int i = 0;

	short *ofmap1 = (short *) calloc(E_C1*F_C1*M_C1*N_C1, sizeof(short));
	short *ofmap2 = (short *) calloc(E_P1*F_P1*C_P1*N_P1, sizeof(short));
	short *ofmap3 = (short *) calloc(E_C2*F_C2*M_C2*N_C2, sizeof(short));
	short *ofmap4 = (short *) calloc(E_P2*F_P2*C_P2*N_P2, sizeof(short));
	short *ofmap5 = (short *) calloc(E_C3*F_C3*M_C3*N_C3, sizeof(short));
	short *ofmap6 = (short *) calloc(E_R1*F_R1*M_R1*N_R1, sizeof(short));

	//Input mapping
	for (i = 0; i<H_C1*W_C1; i++) {
		ifmap[i] = (short)(data[data_set][i] * pow(2, SCALE - 1));
	}

	/////////////////////////////
	//         Layer #1       //
	////////////////////////////
	convolution(ofmap1, ifmap, fmap1, N_C1, C_C1, M_C1, F_C1, E_C1, R_C1, S_C1, H_C1, W_C1, U_C1);
	bias(ofmap1, ofmap1, bias1, N_C1, M_C1, E_C1, F_C1);
	pool(ofmap2, ofmap1, E_C1, F_C1, M_C1);

	/////////////////////////////
	//         Layer #2       //
	////////////////////////////
	convolution(ofmap3, ofmap2, fmap2, N_C2, C_C2, M_C2, F_C2, E_C2, R_C2, S_C2, H_C2, W_C2, U_C2);
	bias(ofmap3, ofmap3, bias2, N_C2, M_C2, E_C2, F_C2);
	pool(ofmap4, ofmap3, E_C2, F_C2, M_C2);

	/////////////////////////////
	//         Layer #3       //
	////////////////////////////
	convolution(ofmap5, ofmap4, fmap3, N_C3, C_C3, M_C3, F_C3, E_C3, R_C3, S_C3, H_C3, W_C3, U_C3);
	bias(ofmap5, ofmap5, bias3, N_C3, M_C3, E_C3, F_C3);
	relu(ofmap6, ofmap5, E_C3, F_C3, M_C3);

	/////////////////////////////
	//         Layer #4       //
	////////////////////////////
	convolution(ofmap, ofmap6, fmap4, N_C4, C_C4, M_C4, F_C4, E_C4, R_C4, S_C4, H_C4, W_C4, U_C4);
	bias(ofmap, ofmap, bias4, N_C4, M_C4, E_C4, F_C4);

	free(ofmap1);
	free(ofmap2);
	free(ofmap3);
	free(ofmap4);
	free(ofmap5);
	free(ofmap6);

	// To this line
	// You are allowed to modify only the inside of this function
	///////////////////////////////////////////////////////////////
}




int main(void) {

	//mnist var////////////////////////////////////////////////////////////
	int data_set = 0, mode_sel = 0;
	float max_val_f = 0;
	short max_val_s = 0;
	int   n = 0, m = 0, e = 0, f = 0, data_index = 0, estimated_label = 0;
	int   ii = 0, kk = 0, i = 0;
	float tmp = 0;

	short *ifmap = 0;
	float *ifmap_f = 0;
	short *ofmap = 0;
	float *ofmap_f = 0;

	float error  = 0;
	float signal = 0;
	float NSRdB = 0;
	//////////////////////////////////////////////////////////////////////

	int x, y;
	int push = 0;
	int pixel_buf_ptr = *(int *)PIXEL_BUF_CTRL_BASE;
	int pixel_ptr, row, col;
	float input_data;

	*(LED_ptr) = 0x00; 	// red LED 초기화
	*(Video_In_DMA_ptr + 3)	= 0x4;				// Enable the video
	printf("start\n");

	while (push != 1) {
		if (*KEY_ptr != 0) {
			*(LED_ptr) = 0xff;
			*(Video_In_DMA_ptr + 3) = 0x0;
			push = 1;

			//draw 28x28 pink box
			for (x = 145; x <= 175; x++) {
				pixel_ptr = pixel_buf_ptr + (105 << 10) + (x << 1);
				*(short *)pixel_ptr = 0xf00f; // set pixel color
				pixel_ptr = pixel_buf_ptr + (135 << 10) + (x << 1);
				*(short *)pixel_ptr = 0xf00f; // set pixel color
			}
			 for (int y = 106; y <= 134; y++) {
				pixel_ptr = pixel_buf_ptr + ( y << 10) + ( 145 << 1);
				*(short *)pixel_ptr = 0xf00f; // set pixel color
				pixel_ptr = pixel_buf_ptr + ( y << 10) + ( 175 << 1);
				*(short *)pixel_ptr = 0xf00f; // set pixel color
			}

			//pixel data
			for (row = 106; row <= 134; row++) {
				for (col = 146; col <= 174; ++col) {
					int i = 0;
					pixel_ptr = pixel_buf_ptr + (row << 10) + (col << 1);
					unsigned char color_val = *(unsigned char *)pixel_ptr;
					//printf("%d  ", data);
					if(color_val<150 && color_val>50) color_val = color_val - 50;
					input_data = 1-(float)color_val/255.0f;
					//printf("%f  ", input_data);
					data[0][i] = input_data;
					printf("%f  ", data[0][i]);
					//printf("%f  ", (1-(float)data/255.0f));
					i++;
				} printf("\n");
			}
		}
	}

//////////////////////////////////////////////////////////
/////                   START                        /////
//////////////////////////////////////////////////////////

	for (mode_sel = 0; mode_sel< _mode; mode_sel++) { // mode_sel = 0 (SW), mode_sel = 1 (HW)

			printf("\r\n");
			if(mode_sel == 0)
				printf("Case 0: Reference \r\n");
			else
				printf("Case 1: Optimization \r\n");

			for (data_set = 0; data_set<_data; data_set++){

				if(mode_sel == 0){
					ifmap_f = (float *)calloc(H_C1*W_C1*C_C1*N_C1,sizeof(float));
					ofmap_f = (float *)calloc(E_C4*F_C4*M_C4*N_C4,sizeof(float));

					cnn_ref(ofmap_f, ifmap_f, data_set);
					/////////////////////////////
					//     Classifiacation     //
					////////////////////////////
					max_val_f = ofmap_f[0];
					data_index = 0;
					for (n = 0; n<N_C4; n++) {
						for (m = 0; m<M_C4; m++) {
							for (e = 0; e<E_C4; e++) {
								for (f = 0; f<F_C4; f++) {

									ofmap_ref[data_set][data_index] = ofmap_f[((n*M_C4 + m)*E_C4 + f)*F_C4 + e];
									//this
									//printf("ofmap_ref[%d][%d] : %d\n", data_set, data_index, ofmap_ref[data_set][data_index]);

									if (ofmap_f[((n*M_C4 + m)*E_C4 + f)*F_C4 + e] >= max_val_f) {
										max_val_f = ofmap_f[((n*M_C4 + m)*E_C4 + f)*F_C4 + e];
										estimated_label = m;
										//printf("max_val_f: %d, estimated_label: %d\n", max_val_f, estimated_label);
									}
									data_index++;
								}
							}
						}
					}
					printf("estimated label: %d (%f) \r\n", estimated_label, max_val_f);

					free(ifmap_f);
					free(ofmap_f);

				}
				/*else{  //if (mode_sel == 1)
					ifmap   = (short *)calloc(H_C1*W_C1*C_C1*N_C1,sizeof(short));
					ofmap   = (short *)calloc(E_C4*F_C4*M_C4*N_C4,sizeof(short));

					cnn_opt(ofmap, ifmap, data_set);

					/////////////////////////////
					//     Classifiacation     //
					////////////////////////////
					max_val_s = ofmap[0];
					data_index = 0;
					for (n = 0; n<N_C4; n++) {
						for (m = 0; m<M_C4; m++) {
							for (e = 0; e<E_C4; e++) {
								for (f = 0; f<F_C4; f++) {

									ofmap_opt[data_set][data_index] = ofmap[((n*M_C4 + m)*E_C4 + f)*F_C4 + e];
									//this
									printf("ofmap_opt[%d][%d] : %d\n", data_set, data_index, ofmap_ref[data_set][data_index]);


									if (ofmap[((n*M_C4 + m)*E_C4 + f)*F_C4 + e] >= max_val_s) {
										max_val_s = ofmap[((n*M_C4 + m)*E_C4 + f)*F_C4 + e];
										estimated_label = m;
									}
									data_index++;
								}
							}
						}
					}*/
					printf("estimated label: %d (%d) \r\n", estimated_label, max_val_s);

					free(ifmap);
					free(ofmap);
				}
			} //data_set

					////////////////////////////////
					// Measure performance
					// (Quantization error)
					////////////////////////////////
					/*if (mode_sel == 1) {
						for (ii = 0; ii<_data; ii++) {
							for (kk = 0; kk<_class; kk++) {
								tmp = (float)(ofmap_opt[ii][kk] * pow(2, -SCALE + 1));
								error += pow(fabsf(ofmap_ref[ii][kk] - tmp), 2);
								signal += pow(fabsf(ofmap_ref[ii][kk]), 2);
							}
						}
						//NSRdB = 10 * log10(error / signal);
						//printf("\r\n");
						//printf("Measure performance: NSR(dB) = %0.3f \r\n", NSRdB);
					}
				}*/ //mode_sel
				printf("END!!!!!");
	}
}
