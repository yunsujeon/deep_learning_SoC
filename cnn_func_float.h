
void relu_f(float *ofmap, float *ifmap, int E, int F, int C)
{
	int c = 0, e = 0, f = 0;

	for (c = 0; c<C; c++)
		for (e = 0; e<E; e++)
			for (f = 0; f<F; f++) {
				ofmap[((c)*E + e)*F + f] = (ifmap[((c)*E + e)*F + f] > 0) ? ifmap[((c)*E + e)*F + f] : 0; //sigmoid 대신쓰는거. O X 가린다
			}
}

void pool_f(float *ofmap, float *ifmap, int E, int F, int C)
{
	int c = 0, e = 0, f = 0, k = 0, l = 0;
	float max = 0;
	int _E = 0, _F = 0;
	_E = E / 2; 
	_F = F / 2; 

	for (c = 0; c<C; c++) { 
		for (e = 0; e<_E; e++) { 
			for (f = 0; f<_F; f++) { 
				max = ifmap[(c*E + (2 * e))*F + (2 * f)];
				for (k = 0; k < 2; k++) {
					for (l = 0; l < 2; l++) {
						max = (max > ifmap[(c*E + 2 * e + k)*F + (2 * f) + l]) ? max : ifmap[(c*E + 2 * e + k)*F + (2 * f) + l];
					} //max pooling
				}
				ofmap[((c)*_E + e)*_F + f] = max;
			}
		}
	}
}

void convolution_f(float *ofmap, float *ifmap, float *fmap, unsigned int N, unsigned int C, unsigned int M, unsigned int E, unsigned int F, unsigned int R, unsigned int S, unsigned int H, unsigned int W, unsigned int U)
{
	unsigned int n = 0, c = 0, m = 0, f = 0, e = 0, r = 0, s = 0;
	float buf = 0;

	// Convolution
	for (n = 0; n < N; n++) { //1
		for (c = 0; c < C; c++) { //1
			for (m = 0; m < M; m++) { // filter 갯수 20
				for (f = 0; f < F; f++) { // output size 24
					for (e = 0; e < E; e++) { // output size 24
						buf = ofmap[((n*M + m)*E + e)*F + f];
						for (r = 0; r < R; r++) { // filter size 5
							for (s = 0; s < S; s++) { // filter size 5 
								buf += ifmap[((n*C + c)*H + e * U + r)*W + f * U + s] * fmap[((m*C + c)*R + r)*S + s]; //input과 filter를 conv
							}
						}
						ofmap[((n*M + m)*E + e)*F + f] = buf;
					}
				}
			}
		}
	}
}
/*
void bias_f(float *ofmap, float *ifmap, float *bias, unsigned int N, unsigned int M, unsigned int E, unsigned int F)
{
	unsigned int n = 0, m = 0, e = 0, f = 0, num = 0;

	// +Bias
	for (n = 0; n<N; n++) // 1
		for (m = 0; m<M; m++) //filter 갯수 20
			for (e = 0; e<E; e++) //outputsize 24
				for (f = 0; f<F; f++) //outputsize 24
					ofmap[((n*M + m)*E + e)*F + f] = ifmap[((n*M + m)*E + e)*F + f] + bias[m]; //bias를 더해준다
					//printf("ofmap%d: %d\n",num, ofmap[((n*M + m)*E + e)*F + f]);
					num ++;
}
*/
void bias_f(float* ofmap, float* ifmap, float* bias, unsigned int N, unsigned int M, unsigned int E, unsigned int F)
{
	unsigned int n = 0, m = 0, e = 0, f = 0, num = 0;

	for (m = 0; m < M; m++) { //filter 갯수 10
		ofmap[m] = ifmap[m] + bias[m]; //bias를 더해준다
	} 
}


void matmul(float* ofmap, float* ifmap, float* fmap, unsigned int N, unsigned int T) {
	unsigned int n = 0, t = 0;
	float buf = 0.0;

	for (n = 0; n < N; n++) { //10까지증가
		for (t = 0; t < T; t++) { //1250까지증가
			buf = ifmap[t] * fmap[(n*T)+t] + buf;
		}
		ofmap[n] = buf;
		buf = 0.0;
	}
}

