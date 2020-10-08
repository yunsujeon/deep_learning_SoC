void relu_f(float *ofmap, float *ifmap, int E, int F, int C)
{
	int c = 0, e = 0, f = 0;

	for (c = 0; c<C; c++)
		for (e = 0; e<E; e++)
			for (f = 0; f<F; f++) {
				ofmap[((c)*E + e)*F + f] = (ifmap[((c)*E + e)*F + f] > 0) ? ifmap[((c)*E + e)*F + f] : 0; //sigmoid ��ž��°�. O X ������
			}
}

void pool_f(float *ofmap, float *ifmap, int E, int F, int C) //��»�����, ��»����� ,���Ͱ���
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

/*
void convolution_f(float* ofmap, float* ifmap, float* fmap, unsigned int N, unsigned int C, unsigned int M, unsigned int E, unsigned int F, unsigned int R, unsigned int S, unsigned int H, unsigned int W, unsigned int U)
{
	unsigned int n = 0, c = 0, m = 0, f = 0, e = 0, r = 0, s = 0;
	float buf = 0;

	// Convolution
	for (n = 0; n < N; n++) { //1
		for (c = 0; c < C; c++) { //���� �β� 1/20
			for (m = 0; m < M; m++) { // filter ���� 20/50
				for (f = 0; f < F; f++) { // output size 24/10
					for (e = 0; e < E; e++) { // output size 24/10
						buf = ofmap[((n * M + m) * E + e) * F + f];
						for (r = 0; r < R; r++) { // filter size 5/3
							for (s = 0; s < S; s++) { // filter size 5/3 
								buf += ifmap[((n * C + c) * H + e * U + r) * W + f * U + s] * fmap[((m * C + c) * R + r) * S + s]; //input�� filter�� conv
							}
						}
						ofmap[((n * M + m) * E + e) * F + f] = buf;
					}
				}
			}
		}
	}
}*/


// convolution_f ( x, x, x, R:���ͻ����� S:���ͻ����� C:���͵β� E:��»����� F:��»����� M:���Ͱ��� H:�Է»����� W:�Է»�����)
void convolution_f(float *ofmap, float *ifmap, float *fmap, unsigned int R, unsigned int S, unsigned int C, unsigned int E, unsigned int F, unsigned int M, unsigned H, unsigned int W)
{
	unsigned int r = 0, s = 0, c = 0, e = 0, f = 0, m = 0, h = 0, w = 0;
	float buf = 0;
	int count = 0, rcount=0;

	for (m = 0; m < M; m++) { //���Ͱ��� 20/50
		for (e = 0; e < E; e++) { //��»����� 24/10
			for (f = 0; f < F; f++) { //��»����� 24/10
				for (c = 0; c < C; c++) { //���͵β� 1/20
					for (r = 0; r < R; r++) { //���ͻ����� 5/3
						for (s = 0; s < S; s++) { //���ͻ����� 5/3
							buf = ifmap[s + (H * r) + (H * W * c) + f + (H * e)] * fmap[(m * S * R * C) + (count % (S * R * C))] + buf;
							count = count + 1;
						}
					}
				}
				ofmap[rcount] = buf;
				rcount++;
				buf = 0;
			}
		}
	}
	printf("\n%d\n", count);
	printf("\n%d\n", rcount);
}

void bias_f(float* ofmap, float* ifmap, float* bias, unsigned int N, unsigned int M, unsigned int E, unsigned int F)
{
	unsigned int n = 0, m = 0, e = 0, f = 0, num = 0;

	for (m = 0; m < M; m++) { //filter ���� 10
		ofmap[m] = ifmap[m] + bias[m]; //bias�� �����ش�
	} 
}


void matmul(float* ofmap, float* ifmap, float* fmap, unsigned int N, unsigned int T) {
	unsigned int n = 0, t = 0;
	float buf = 0.0;
	int count = 0;
	for (n = 0; n < N; n++) { //0-9��������
		for (t = 0; t < T; t++) { //0-1249���� ����
			buf = ifmap[t] * fmap[(count%T)*N+(count/T)] + buf;
			count += 1;
		}
		ofmap[n] = buf;
		buf = 0.0;
	}
}






/*
void bias_f(float *ofmap, float *ifmap, float *bias, unsigned int N, unsigned int M, unsigned int E, unsigned int F)
{
	unsigned int n = 0, m = 0, e = 0, f = 0, num = 0;

	// +Bias
	for (n = 0; n<N; n++) // 1
		for (m = 0; m<M; m++) //filter ���� 20
			for (e = 0; e<E; e++) //outputsize 24
				for (f = 0; f<F; f++) //outputsize 24
					ofmap[((n*M + m)*E + e)*F + f] = ifmap[((n*M + m)*E + e)*F + f] + bias[m]; //bias�� �����ش�
					//printf("ofmap%d: %d\n",num, ofmap[((n*M + m)*E + e)*F + f]);
					num ++;
}
*/