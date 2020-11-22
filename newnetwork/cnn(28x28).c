#include<stdio.h>
#include"param.h"


#define IMG_DMNIN 28
#define IMG_CHANNELS 1

//layer 1
#define C1_N_CHAN 1
#define C1_X_DMNIN 28
#define C1_W_DMNIN 5
#define C1_OUT_DMNIN 24
#define C1_N_FILTERS 10

#define A1_ROWS 24
#define A1_COLS 24
#define P1_SIZE 24
#define P1_DOWNSIZE 12

//layer 2
#define C2_N_CHAN 10
#define C2_X_DMNIN 12
#define C2_W_DMNIN 5
#define C2_OUT_DMNIN 8
#define C2_N_FILTERS 20

#define A2_ROWS 8
#define A2_COLS 8
#define P2_SIZE 8
#define P2_DOWNSIZE 4

//fully
#define FLAT_SIZE 320
#define F_ROWS 10
#define F_COLS 320

//const uint16_t FLAT_VEC_SZ = 400;
//const uint8_t F1_ROWS = 120;
//const uint16_t F1_COLS = 400;

//const uint8_t F2_ROWS = 84;
//const uint8_t F2_COLS = 120;


float w123[3] = { 0.12432, 0.876234, 0.23476 };
float res[3] = { 0 };
void fc(float w[]) {
	for (int i = 0; i < 3; i++) {
		res[i] = w[i] * i;
	}
}

void convolution(
	int N_FILTERS, int N_CHAN, int OUT_DMNIN, int W_DMNIN, int X_DMNIN,
	float x[][28][28], float w[][10][5][5], float conv_out[][24][24]){	
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
	int N_FILTERS, int N_CHAN, int OUT_DMNIN, int W_DMNIN, int X_DMNIN,	float x[][12][12], float w[][20][5][5], float conv_out[][8][8]) {
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

void relu(
	int N_FILTERS, int A_ROWS, int A_COLS,
	float relu_in[][24][24],
	float relu_out[][24][24]){
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
	float relu_in[][8][8],
	float relu_out[][8][8]) {
	for (int f = 0; f < N_FILTERS; f++) {
		for (int i = 0; i < A_ROWS; i++) {
			for (int j = 0; j < A_COLS; j++) {
				relu_out[f][i][j] = (relu_in[f][i][j] > 0) ? relu_in[f][i][j] : 0;
				//printf("relu_out[%d][%d][%d] = %f\n", f,i,j, relu_out[f][i][j]);
			}
		}
	}
}

void max_pooling(
	int N_FILTERS, int P_SIZE, int P_DOWNSIZE,
	float max_in[][24][24],
	float max_out[][12][12]){
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
	float max_in[][8][8],
	float max_out[][4][4]) {
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

void flatten(float flat_in[][4][4], float flat_out[]){
	for(int f=0, t=0; f<20; f++){
		for(int i=0; i<4; i++){
			for(int j=0; j<4; j++){
				flat_out[t++] = flat_in[f][i][j];
			}
		}
	}		
}


void fully_connect(int FULLY_ROWS, int FULLY_COLS, float fully_in[], float w[], float fully_out[]){
	float buf;
	for(int i=0; i<FULLY_ROWS; i++){
		for(int j=0; j<FULLY_COLS; j++){
			buf = w[j+i*320]*fully_in[j];
			fully_out[i] += buf;
			//fully_out[i] += w[j*10+i]*fully_in[j];
			//printf("fully_in[%d]=%f\n",j,fully_in[j]);
			//printf("weight[%d]=%f\n",i*10+j,w[i*10+j]);
			//printf("buf[%d]=%f\n",j+i*320, buf);
		} 
		//printf("fully_out[%d]=%f\n",i,fully_out[i]);
	}
}
/*
void fully_connect(int FULLY_ROWS, int FULLY_COLS, float fully_in[], float w[], float fully_out[]) {
	unsigned int n = 0, t = 0;
	float buf = 0.0;
	int count = 0;
	for (n = 0; n < 10; n++) {
		for (t = 0; t < 320; t++) {
			buf = fully_in[t] * w[(count % 320) * 10 + (count / 320)] + buf;
			count += 1;
		}
		fully_out[n] = buf;
		buf = 0.0;
	}
}
*/

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
	float flat_out[FLAT_SIZE];
	float result[F_ROWS];

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
	for(int t=0; t<FLAT_SIZE; t++){
		flat_out[t] = 0;
	}
	
	//result 초기화
	for(int r=0; r<10; r++){
		result[r]=0;
	}

	fc(w123);
	/*
	convolution(C1_N_FILTERS, C1_N_CHAN, C1_OUT_DMNIN, C1_W_DMNIN, C1_X_DMNIN, image, weights_C1, conv1_out);
	relu(C1_N_FILTERS, A1_ROWS, A1_COLS, conv1_out, relu1_out);
	max_pooling(C1_N_FILTERS, P1_SIZE, P1_DOWNSIZE, relu1_out, max1_out);
	
	convolution(C2_N_FILTERS, C2_N_CHAN, C2_OUT_DMNIN, C2_W_DMNIN, C2_X_DMNIN, max1_out, weights_C2, conv2_out);
	relu(C2_N_FILTERS, A2_ROWS, A2_COLS, conv2_out, relu2_out);
	max_pooling(C2_N_FILTERS, P2_SIZE, P2_DOWNSIZE, relu2_out, max2_out);
	*/
	convolution(C1_N_FILTERS, C1_N_CHAN, C1_OUT_DMNIN, C1_W_DMNIN, C1_X_DMNIN, image, weights_C1, conv1_out);
	max_pooling(C1_N_FILTERS, P1_SIZE, P1_DOWNSIZE, conv1_out, max1_out);

	convolution2(C2_N_FILTERS, C2_N_CHAN, C2_OUT_DMNIN, C2_W_DMNIN, C2_X_DMNIN, max1_out, weights_C2, conv2_out);
	max_pooling2(C2_N_FILTERS, P2_SIZE, P2_DOWNSIZE, conv2_out, max2_out);

	flatten(max2_out, flat_out);
	fully_connect(F_ROWS, F_COLS, flat_out, weights_C3, result);
	
	for(int i=0; i<10; i++){
		result[i] = result[i] + bias[i];
		printf("result[%d]=%f\n", i, result[i]);
	}

		
	for(int t=0; t<FLAT_SIZE; t++){
		//printf("flat_out[%d]=%f\n",t, flat_out[t]);
	}
	
	/*
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
