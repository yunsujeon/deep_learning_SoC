void relu(short *ofmap, short *ifmap, int E, int F, int C)
{//activation function(어떤 값을 보내게 될때 일정값 이상이면 active)으로
//rectified linear unit. sigmoid 대신에 쓰는것 0보다 작을땐 non-active 클때는 그값을 그대로반환
	int c = 0, e = 0, f = 0;

	for (c = 0; c<C; c++)
		for (e = 0; e<E; e++)
			for (f = 0; f<F; f++) {
				ofmap[((c)*E + e)*F + f] = (ifmap[((c)*E + e)*F + f] > 0) ? ifmap[((c)*E + e)*F + f] : 0;
			}
}

void pool(short *ofmap, short *ifmap, int E, int F, int C) //convolution을 거쳐서 나온 activation maps가 있을때 이를 이루는 convolution layer를 resizing 하여 새로운 layer를 얻는것
{//convolution으로 feature가 많아짐으로 인한 overfitting을 방지하기 위해 pooling 해준다. mean pooling / max pooling
	int c = 0, e = 0, f = 0, k = 0, l = 0;
	short max = 0;
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
					}
				}
				ofmap[((c)*_E + e)*_F + f] = max;
			}
		}
	}
}

void convolution(short *ofmap, short *ifmap, short *fmap, unsigned int N, unsigned int C, unsigned int M, unsigned int E, unsigned int F, unsigned int R, unsigned int S, unsigned int H, unsigned int W, unsigned int U)
{
	unsigned int n = 0, c = 0, m = 0, f = 0, e = 0, r = 0, s = 0;
	int buf = 0;

	// Convolution
	for (n = 0; n < N; n++) {
		for (c = 0; c < C; c++) {
			for (m = 0; m < M; m++) {
				for (f = 0; f < F; f++) {
					for (e = 0; e < E; e++) {
						buf = (int)(ofmap[((n*M + m)*E + e)*F + f] * pow(2, SCALE));
						for (r = 0; r < R; r++) {
							for (s = 0; s < S; s++) {
								//ofmap[((n*M + m)*E + e)*F + f] += ifmap[((n*C + c)*H + e * U + r)*W + f * U + s] * fmap[((m*C + c)*R + r)*S + s];
								buf += ifmap[((n*C + c)*H + e * U + r)*W + f * U + s] * fmap[((m*C + c)*R + r)*S + s];
							}
						}
						ofmap[((n*M + m)*E + e)*F + f] = (short)(buf * pow(2, -SCALE));
					}
				}
			}
		}
	}
}

void bias(short *ofmap, short *ifmap, short *bias, unsigned int N, unsigned int M, unsigned int E, unsigned int F)
{//이 값이 높을수록 분류의 기준이 엄격하다는 것을 의미
//높을수록 모델이 간단해지는 경향(변수가 적고 일반화) underfitting위험
//작을수록 데이터의 허용범위가 넓어져 학습데이터에만 잘 들어맞는 모델이 만들어짐
	unsigned int n = 0, m = 0, e = 0, f = 0;

	// +Bias
	for (n = 0; n<N; n++)
		for (m = 0; m<M; m++)
			for (e = 0; e<E; e++)
				for (f = 0; f<F; f++)
					ofmap[((n*M + m)*E + e)*F + f] = ifmap[((n*M + m)*E + e)*F + f] + bias[m];
}
