#include<stdio.h>
#include"param.h"

#define IMG_DMNIN 32
#define IMG_CHANNELS 1

//layer 1
#define C1_N_CHAN 1
#define C1_X_DMNIN 32
#define C1_W_DMNIN 5
#define C1_OUT_DMNIN 28
#define C1_N_FILTERS 6

#define A1_ROWS 28
#define A1_COLS 28
#define P1_SIZE 28
#define P1_DOWNSIZE 14

//layer 2
#define C2_N_CHAN 6
#define C2_X_DMNIN 14
#define C2_W_DMNIN 5
#define C2_OUT_DMNIN 10
#define C2_N_FILTERS 16

#define A2_ROWS 10
#define A2_COLS 10
#define P2_SIZE 10
#define P2_DOWNSIZE 5

//fully
#define FLAT_VEC_SZ 400
#define F1_ROWS 120
#define F1_COLS 400
#define F2_ROWS 84
#define F2_COLS 120
#define F3_ROWS 10
#define F3_COLS 84

//softmax
#define SFMX_SIZE 10
#define SFMX_RES 400



void convolution1(
	int N_FILTERS, int N_CHAN, int OUT_DMNIN, int W_DMNIN, int X_DMNIN,
	//float *x, const float *w, float *conv_out){	
	const float x[][32][32], const float w[][6][5][5], float conv_out[][28][28]){	
	float mul;
	for(int f=0; f<N_FILTERS; f++){
		for(int ch=0; ch<N_CHAN; ch++){
			for(int i=0; i<OUT_DMNIN; i++){
				for(int j=0; j<OUT_DMNIN; j++){
					for(int m=0; m<W_DMNIN; m++){
						for(int n=0; n<W_DMNIN; n++){
							mul = x[ch][m+i][n+j] * w[ch][f][m][n];
							conv_out[f][i][j] = conv_out[f][i][j]+mul;
							//printf("mul[%d][%d]:%f\n",m,n,mul);
							//conv_out[f][i][j] += x[ch][m+i][n+j] * w[ch][f][m][n];
						}
          } //printf("conv_out[%d][%d][%d]=%f\n",f,i,j,conv_out[f][i][j]); conv_out[f][i][j] = 0;
        }
      }
    }
  }
}

void convolution2(
	int N_FILTERS, int N_CHAN, int OUT_DMNIN, int W_DMNIN, int X_DMNIN,
	//float *x, const float *w, float *conv_out){	
	const float x[][14][14], const float w[][16][5][5], float conv_out[][10][10]) {
	float mul;
	for (int f = 0; f < N_FILTERS; f++) {
		for (int ch = 0; ch < N_CHAN; ch++) {
			for (int i = 0; i < OUT_DMNIN; i++) {
				for (int j = 0; j < OUT_DMNIN; j++) {
					for (int m = 0; m < W_DMNIN; m++) {
						for (int n = 0; n < W_DMNIN; n++) {
							mul = x[ch][m + i][n + j] * w[ch][f][m][n];
							conv_out[f][i][j] = conv_out[f][i][j] + mul;
							//printf("mul[%d][%d]:%f\n",m,n,mul);
							//conv_out[f][i][j] += x[ch][m+i][n+j] * w[ch][f][m][n];
						}
					} //printf("conv_out[%d][%d][%d]=%f\n",f,i,j,conv_out[f][i][j]); conv_out[f][i][j] = 0;
				}
			}
		}
	}
}

void relu1(
	int N_FILTERS, int A_ROWS, int A_COLS,
	float relu_in[][28][28],
	float relu_out[][28][28]){
	for(int f=0; f<N_FILTERS; f++){
		for(int i=0; i<A_ROWS; i++){
			for(int j=0; j<A_COLS; j++){
				relu_out[f][i][j] = (relu_in[f][i][j]> 0)? relu_in[f][i][j] :0;
				//printf("relu_out[%d][%d][%d] = %f\n", f,i,j, relu_out[f][i][j]);
			}
		}
	}
}

void relu2(
	int N_FILTERS, int A_ROWS, int A_COLS,
	float relu_in[][10][10],
	float relu_out[][10][10]) {
	for (int f = 0; f < N_FILTERS; f++) {
		for (int i = 0; i < A_ROWS; i++) {
			for (int j = 0; j < A_COLS; j++) {
				relu_out[f][i][j] = (relu_in[f][i][j] > 0) ? relu_in[f][i][j] : 0;
				//printf("relu_out[%d][%d][%d] = %f\n", f,i,j, relu_out[f][i][j]);
			}
		}
	}
}

void max_pooling1(
	int N_FILTERS, int P_SIZE, int P_DOWNSIZE,
	float max_in[][28][28],
	float max_out[][14][14]){
	float max_num=0;
	for(int f=0; f<N_FILTERS; f++){
		for(int i=0; i<P_DOWNSIZE; i++){
			for(int j=0; j<P_DOWNSIZE; j++){
				for(int m=0; m<2; m++){
					for(int n=0; n<2; n++){
						if(max_in[f][m+(i*2)][n+(j*2)] > max_num){
							max_num = max_in[f][m+(i*2)][n+(j*2)];
							//printf("max_in[%d][%d][%d]=%f\n", f, m+i*2, n+j*2, max_in[f][m+(i*2)][n+(j*2)]);
						}
							//printf("max_num1 = %d\n", max_num);
					}
				} max_out[f][i][j] = max_num;
				//printf("max_out[%d][%d][%d]=%f\n",f,i,j,max_out[f][i][j]);
				max_num=0;
			}
		}
	 }
}

void max_pooling2(
	int N_FILTERS, int P_SIZE, int P_DOWNSIZE,
	float max_in[][10][10],
	float max_out[][5][5]) {
	float max_num = 0;
	for (int f = 0; f < N_FILTERS; f++) {
		for (int i = 0; i < P_DOWNSIZE; i++) {
			for (int j = 0; j < P_DOWNSIZE; j++) {
				for (int m = 0; m < 2; m++) {
					for (int n = 0; n < 2; n++) {
						if (max_in[f][m + (i * 2)][n + (j * 2)] > max_num) {
							max_num = max_in[f][m + (i * 2)][n + (j * 2)];
							//printf("max_in[%d][%d][%d]=%f\n", f, m+i*2, n+j*2, max_in[f][m+(i*2)][n+(j*2)]);
						}
						//printf("max_num1 = %d\n", max_num);
					}
				} max_out[f][i][j] = max_num;
				//printf("max_out[%d][%d][%d]=%f\n",f,i,j,max_out[f][i][j]);
				max_num = 0;
			}
		}
	}
}

void flatten(float flat_in[][5][5], float flat_out[FLAT_VEC_SZ]){
	for(int f=0, t=0; f<16; f++){
		for(int i=0; i<5; i++){
			for(int j=0; j<5; j++){
				flat_out[t++] = flat_in[f][i][j];
			}
		}
	}		
}

void fully_connect1(int F_ROWS, int F_COLS, float fully_in[], const float w[][400], float fully_out[]){
	for(int i=0; i<F_ROWS; i++){
		for(int j=0; j<F_COLS; j++){
			fully_out[i] += w[i][j]*fully_in[j];
			//printf("weight=[%d][%d] = %f\n",i,j,w[i][j]);
			//printf("fully_out=[%d] = %f\n",i,fully_out[i]);
		}
	}
}

void fully_connect2(int F_ROWS, int F_COLS, float fully_in[], const float w[][120], float fully_out[]) {
	for (int i = 0; i < F_ROWS; i++) {
		for (int j = 0; j < F_COLS; j++) {
			fully_out[i] += w[i][j] * fully_in[j];
			//printf("weight=[%d][%d] = %f\n",i,j,w[i][j]);
			//printf("fully_out=[%d] = %f\n",i,fully_out[i]);
		}
	}
}

void fully_connect3(int F_ROWS, int F_COLS, float fully_in[], const float w[][84], float fully_out[]) {
	for (int i = 0; i < F_ROWS; i++) {
		for (int j = 0; j < F_COLS; j++) {
			fully_out[i] += w[i][j] * fully_in[j];
			//printf("weight=[%d][%d] = %f\n",i,j,w[i][j]);
			//printf("fully_out=[%d] = %f\n",i,fully_out[i]);
		}
	}
}

int main(){

	//FILE *fout;
	//int correct=0;

	//fout = fopen("result_y.dat","w");
	//if(fout==NULL) printf("file result open error\n");

	float conv1_out[C1_N_FILTERS][C1_OUT_DMNIN][C1_OUT_DMNIN];
	float relu1_out[C1_N_FILTERS][A1_ROWS][A1_COLS];
	float max1_out[C1_N_FILTERS][P1_DOWNSIZE][P1_DOWNSIZE];
	float conv2_out[C2_N_FILTERS][C2_OUT_DMNIN][C2_OUT_DMNIN];
	float relu2_out[C2_N_FILTERS][A2_ROWS][A2_COLS];
	float max2_out[C2_N_FILTERS][P2_DOWNSIZE][P2_DOWNSIZE];
	float flat_out[FLAT_VEC_SZ];
	float fully1_out[F1_ROWS];
	float fully2_out[F2_ROWS];
	float fully3_out[F3_ROWS];
	
	//layer1 변수들 초기화
	for(int f=0; f<C1_N_FILTERS; f++){
		for(int i=0; i<C1_OUT_DMNIN; i++){
			for(int j=0; j<C1_OUT_DMNIN; j++){
				conv1_out[f][i][j] = 0;
				relu1_out[f][i][j] = 0;
				max1_out[f][i/2][j/2]=0;
			}
		}
	}
	
	//layer2 변수들 초기화
	for(int f=0; f<C2_N_FILTERS; f++){
		for(int i=0; i<C2_OUT_DMNIN; i++){
			for(int j=0; j<C2_OUT_DMNIN; j++){
				conv2_out[f][i][j] = 0;
				relu2_out[f][i][j] = 0;
				max2_out[f][i/2][j/2]=0;
			}
		}
	}
	
	//flatten 변수 초기화
	for(int t=0; t<FLAT_VEC_SZ; t++){
		flat_out[t] = 0;
	}
	
	//fully1 변수 초기화
	for(int t=0; t<F1_ROWS; t++){
		fully1_out[t] = 0;
	}
	
	//fully2 변수 초기화
	for(int t=0; t<F2_ROWS; t++){
		fully2_out[t] = 0;
	}
	
	//fully3 변수 초기화
	for(int t=0; t<F3_ROWS; t++){
		fully3_out[t] = 0;
	}
	
	
	convolution1(C1_N_FILTERS, C1_N_CHAN, C1_OUT_DMNIN, C1_W_DMNIN, C1_X_DMNIN, image, weights_C1, conv1_out);
	relu1(C1_N_FILTERS, A1_ROWS, A1_COLS, conv1_out, relu1_out);
	max_pooling1(C1_N_FILTERS, P1_SIZE, P1_DOWNSIZE, relu1_out, max1_out);
	
	convolution2(C2_N_FILTERS, C2_N_CHAN, C2_OUT_DMNIN, C2_W_DMNIN, C2_X_DMNIN, max1_out, weights_C2, conv2_out);
	relu2(C2_N_FILTERS, A2_ROWS, A2_COLS, conv2_out, relu2_out);
	max_pooling2(C2_N_FILTERS, P2_SIZE, P2_DOWNSIZE, relu2_out, max2_out);
	
	flatten(max2_out, flat_out);
	fully_connect1(F1_ROWS, F1_COLS, flat_out, weights_F1, fully1_out);
	fully_connect2(F2_ROWS, F2_COLS, fully1_out, weights_F2, fully2_out);
	fully_connect3(F3_ROWS, F3_COLS, fully2_out, weights_F3, fully3_out);	
	
	printf("\n");
	//fully3 test
	for(int t=0; t<F3_ROWS; t++){
		printf("   fully3_out[%d] = %f\n",t, fully3_out[t]);
	}	
	
	float MAXRES = -100.0;
	int MAXIDX = -100;
	for (int i = 0; i < 10; i++) {
		if (MAXRES <= fully3_out[i]) {
			MAXRES = fully3_out[i];
			MAXIDX = i;
		}
	}
	printf("   estimated labed : %d", MAXIDX);


	/*
	//fully2 test
	for(int t=0; t<F2_ROWS; t++){
		printf("fully2_out[%d] = %f\n",t, fully2_out[t]);
	}
	
	
	//fully1 test
	for(int t=0; t<F1_ROWS; t++){
		printf("fully1_out[%d] = %f\n",t, fully1_out[t]);
	}*/
	
	
	//flatten test
	//for(int t=0; t<FLAT_VEC_SZ; t++){
	//	printf("flat_out[%d]=%f\n",t, flat_out[t]);
	//}
	
	/*
	//max_pooling2 test
	for(int f=0; f<C2_N_FILTERS; f++){
		for(int i=0; i<P2_DOWNSIZE; i++){
			for(int j=0; j<P2_DOWNSIZE; j++){
				printf("max2_out[%d][%d][%d]=%f\n",f,i,j,max2_out[f][i][j]);
			}
		}
	}*/
		

//  일치확인
/*for(int k=0; k<6; k++){
  for(int i=0; i<28; i++){
    for(int j=0; j<28; j++){
      fscanf(fin_y, "%d ", &num);
      file_y[k][i][j] = num;
      if(file_y[k][i][j] == c[k][i][j] ) correct ++;
    }
  }
}  printf("%d\n",correct);
*/

  return 0;
}
