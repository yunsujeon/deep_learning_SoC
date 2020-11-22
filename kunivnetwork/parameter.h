/******************************************************************************
Copyright (c) 2017 SoC Design Laboratory, Konkuk University, South Korea
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met: redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer;
redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution;
neither the name of the copyright holders nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Jooho Wang (joohowang@konkuk.ac.kr)

Revision History
2017.11.14: Started by Jooho Wang
*******************************************************************************/
//
#define _mode  1
#define _data  1 //0�� ����ҰŸ� 1�εֵ��ȴ�
#define _class 10 //mnist ���ڴϱ� 10���� ����� ��Ÿ���Եȴ�.
#define SCALE  15

//Convlayer #1 parameter
#define N_C1 1 
#define M_C1 10 // ���Ͱ���
#define C_C1 1 //���� �β�
#define E_C1 24 // ��»�����
#define F_C1 24 // ��»�����
#define R_C1 5  // ���ͻ�����
#define S_C1 5  // ���ͻ�����
#define H_C1 28 // �Է»�����
#define W_C1 28 // �Է»�����
#define U_C1 1

//Pool #1 parameter
#define N_P1 1
#define E_P1 12 //�������°� �ƴ� �޸��Ҵ��� ���� �̸����
#define F_P1 12
#define C_P1 10 
#define M_P1 10

//Convlayer #2 parameter
#define N_C2 1
#define M_C2 20 //���Ͱ���
#define C_C2 10 //���� �β�
#define E_C2 8 //��»�����
#define F_C2 8 //��»�����
#define R_C2 5 //���ͻ�����
#define S_C2 5 //���ͻ�����
#define H_C2 12 //�Է»�����
#define W_C2 12 //�Է»�����
#define U_C2 1

//Pool #2 parameter
#define N_P2 1
#define E_P2 4
#define F_P2 4
#define C_P2 20
#define M_P2 20

//Convlayer #3 parameter
#define N_C3 1
#define M_C3 10
#define C_C3 20
#define E_C3 1
#define F_C3 1
#define R_C3 4
#define S_C3 4
#define H_C3 4
#define W_C3 4
#define U_C3 1

//ReLU #1 parameter
#define N_R1 1
#define M_R1 50
#define C_R1 20
#define E_R1 10
#define F_R1 10
#define R_R1 4
#define S_R1 4
#define H_R1 4
#define W_R1 4
#define U_R1 1


//Matmul #1 parameter
#define N_M1 10
#define T_M1 320