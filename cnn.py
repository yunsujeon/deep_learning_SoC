# #이진 으로 바꾸기
#
# num1 = -0.1701116
# num2 = 0.26553392
# num3 = -0.014648308
# num4 = 0.0
# num5 = -7.125691e-06
# num6 = -0.00028364424
# num7 = 0.00045066405
# num8 = 7.242351368
#
# # num1 = 0.1*0.96
# # num2 = 0.2*0.96
# # num3 = 0.3*0.96
# # num4 = 0.4*0.96
# # num5 = 0.5*0.96
# # num6 = 0.6*0.96
# # num7 = 0.7*0.96
# # num8 = 0.8*0.96
# num9 = 0.9*0.96
#
# #포맷재지정
# num1 = format(num1, ".6f")
# num2 = format(num2, ".6f")
# num3 = format(num3, ".6f")
# num4 = format(num4, ".6f")
# num5 = format(num5, ".6f")
# num6 = format(num6, ".6f")
# num7 = format(num7, ".6f")
# num8 = format(num8, ".6f")
# num9 = format(num9, ".6f")
#
# def float_bin(number, places):
#     whole, trash = str(number).split(".")
#     number = float(number)
#     dec = abs(number)
#     whole = int(whole)
#     #dec = int(dec)
#     if (whole==0):
#         res = "0."
#     else:
#         res = bin(whole).lstrip("0b") + "."
#
#     for x in range(places):
#         fdec = dec*2
#         fdec = format(fdec, ".6f")
#         whole, dec = fdec.split(".")
#         dec = "0."+dec
#         dec = float(dec)
#         res += whole
#     return res
#
# # def decimal_converter(num):
# #     while num > 1:
# #         num /= 10
# #     if( num == 0 ):
# #         return 0.0
# #     return num
#
#
#
# p = int(16)
#
# num1 = (float_bin(num1, places=p))
# num2 = (float_bin(num2, places=p))
# num3 = (float_bin(num3, places=p))
# num4 = (float_bin(num4, places=p))
# num5 = (float_bin(num5, places=p))
# num6 = (float_bin(num6, places=p))
# num7 = (float_bin(num7, places=p))
# num8 = (float_bin(num8, places=p))
# num9 = (float_bin(num9, places=p))
#
# print(num1)
# print(num2)
# print(num3)
# print(num4)
# print(num5)
# print(num6)
# print(num7)
# print(num8)
# print(num9)
#
# a=10
# b=10


###########################################################   NN (1회)   #############################################################

import tensorflow as tf
import numpy as np
#---------------------------------------------------------------------------------------------------- 1. MNIST 데이터를 가져옵니다.

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#---------------------------------------------------------------------------------------------------- 2. 모델을 생성합니다.
x = tf.placeholder(tf.float32, [None, 784])

# 모델 파라미터
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# softmax를 사용한 모델을 생성
y_model = tf.matmul(x, W) + b #행렬 곱연산과 bias 더해주기

#---------------------------------------------------------------------------------------------------- 3. loss와 optimizer를 정의합니다.
y = tf.placeholder(tf.float32, [None, 10])  # 크기 10인 MNIST의 라벨 데이터 (숫자가 열개니깐)

#cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(tf.nn.softmax(y_model)), reduction_indices=1))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_model))

# Gradient Descent - Backpropagation 기법으로 최적화
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost) # learning_rate = 0.01

#---------------------------------------------------------------------------------------------------- 4. 훈련을 위한 세션 시작
sess = tf.Session()
sess.run(tf.global_variables_initializer()) # 변수 초기화

for epoch in range(35): # 훈련을 35번 반복
    avg_cost = 0.

    # 1번 훈련시 전체 훈련 데이터를 사용하려면 100개씩 몇번 가져와야 하는지 계산하여 반복
    total_batch = int(mnist.train.num_examples / 100)
    for i in range(total_batch):
        # 전체 훈련 데이터(mnist.train)에서 100개씩 데이터를 가져옵니다.
        # (100, 784) (100, 10)
        batch_xs, batch_ys = mnist.train.next_batch(100)

        # optimizer와 cost 오퍼레이션을 실행합니다.
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})

        # 현재까지 평균 손실(loss)를 누적합니다.
        avg_cost += c / total_batch

    # 훈련 1번 끝날때 마다 중간 결과를 출력
    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

print("최적화 완료")

count = 0
count1 = 0

#---------------------------------------------------------------------------------------------------- 5. 정확도 측정
# 라벨값 y와 모델로 계산된 값 y_model이 똑같이 같은 인덱스가 제일 크다고 하는지 검사
# ( tf.argmax 함수가 배열에서 가장 큰 값을 가리키는 인덱스를 리턴합니다.. )
# 결과적으로 correct_prediction는 True 또는 False 값의 리스트를 가지게 됨

correct_prediction = tf.equal(tf.argmax(y_model, 1), tf.argmax(y, 1))

# tf.cast 함수를 사용하여 True 또는 False를 실수 1 또는 0으로 변환
# 전체 데이터가 일치한다면 모든 값이 1이며 평균인 accuracy는 1이 되어야 합니다.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 정확도 측정을 위해서 훈련 데이터(mnist.train) 대신에 별도의 테스트 데이터(mnist.test)를 사용해야 합니다.
print("Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

# image_file01 = open("C:/Users/dbstn/Desktop/numdata/num0_x1.txt","r")
# image_file02 = open("C:/Users/dbstn/Desktop/numdata/num0_x2.txt","r")
# image_file11 = open("C:/Users/dbstn/Desktop/numdata/num1_o1.txt","r")
# image_file21 = open("C:/Users/dbstn/Desktop/numdata/num2_o1.txt","r")
# image_file22 = open("C:/Users/dbstn/Desktop/numdata/num2_o2.txt","r")
# image_file31 = open("C:/Users/dbstn/Desktop/numdata/num3_o1.txt","r")
# image_file32 = open("C:/Users/dbstn/Desktop/numdata/num3_o2.txt","r")
# image_file41 = open("C:/Users/dbstn/Desktop/numdata/num4_o1.txt","r")
# image_file42 = open("C:/Users/dbstn/Desktop/numdata/num4_o2.txt","r")
# image_file51 = open("C:/Users/dbstn/Desktop/numdata/num5_o1.txt","r")
# image_file52 = open("C:/Users/dbstn/Desktop/numdata/num5_o2.txt","r")
# image_file61 = open("C:/Users/dbstn/Desktop/numdata/num6_o1.txt","r")
# image_file62 = open("C:/Users/dbstn/Desktop/numdata/num6_o2.txt","r")
# image_file71 = open("C:/Users/dbstn/Desktop/numdata/num7_x1.txt","r")
# image_file81 = open("C:/Users/dbstn/Desktop/numdata/num8_o1.txt","r")
# image_file91 = open("C:/Users/dbstn/Desktop/numdata/num9_o1.txt","r")
#
# image01 = image_file01.read().split(', ')
# image02 = image_file02.read().split(', ')
# image11 = image_file11.read().split(', ')
# image21 = image_file21.read().split(', ')
# image22 = image_file22.read().split(', ')
# image31 = image_file31.read().split(', ')
# image32 = image_file32.read().split(', ')
# image41 = image_file41.read().split(', ')
# image42 = image_file42.read().split(', ')
# image51 = image_file51.read().split(', ')
# image52 = image_file52.read().split(', ')
# image61 = image_file61.read().split(', ')
# image62 = image_file62.read().split(', ')
# image71 = image_file71.read().split(', ')
# image81 = image_file81.read().split(', ')
# image91 = image_file91.read().split(', ')
#
# image01 = list(map(float, image01))
# image02 = list(map(float, image02))
# image11 = list(map(float, image11))
# image21 = list(map(float, image21))
# image22 = list(map(float, image22))
# image31 = list(map(float, image31))
# image32 = list(map(float, image32))
# image41 = list(map(float, image41))
# image42 = list(map(float, image42))
# image51 = list(map(float, image51))
# image52 = list(map(float, image52))
# image61 = list(map(float, image61))
# image62 = list(map(float, image62))
# image71 = list(map(float, image71))
# image81 = list(map(float, image81))
# image91 = list(map(float, image91))
#
# classification = (sess.run(tf.argmax(y_model, 1), feed_dict={x : [image01]}))
# print('Neural Network predicted', classification[0])
# classification = (sess.run(tf.argmax(y_model, 1), feed_dict={x : [image02]}))
# print('Neural Network predicted', classification[0])
# classification = (sess.run(tf.argmax(y_model, 1), feed_dict={x : [image11]}))
# print('Neural Network predicted', classification[0])
# classification = (sess.run(tf.argmax(y_model, 1), feed_dict={x : [image21]}))
# print('Neural Network predicted', classification[0])
# classification = (sess.run(tf.argmax(y_model, 1), feed_dict={x : [image22]}))
# print('Neural Network predicted', classification[0])
# classification = (sess.run(tf.argmax(y_model, 1), feed_dict={x : [image31]}))
# print('Neural Network predicted', classification[0])
# classification = (sess.run(tf.argmax(y_model, 1), feed_dict={x : [image32]}))
# print('Neural Network predicted', classification[0])
# classification = (sess.run(tf.argmax(y_model, 1), feed_dict={x : [image41]}))
# print('Neural Network predicted', classification[0])
# classification = (sess.run(tf.argmax(y_model, 1), feed_dict={x : [image42]}))
# print('Neural Network predicted', classification[0])
# classification = (sess.run(tf.argmax(y_model, 1), feed_dict={x : [image51]}))
# print('Neural Network predicted', classification[0])
# classification = (sess.run(tf.argmax(y_model, 1), feed_dict={x : [image52]}))
# print('Neural Network predicted', classification[0])
# classification = (sess.run(tf.argmax(y_model, 1), feed_dict={x : [image61]}))
# print('Neural Network predicted', classification[0])
# classification = (sess.run(tf.argmax(y_model, 1), feed_dict={x : [image62]}))
# print('Neural Network predicted', classification[0])
# classification = (sess.run(tf.argmax(y_model, 1), feed_dict={x : [image71]}))
# print('Neural Network predicted', classification[0])
# classification = (sess.run(tf.argmax(y_model, 1), feed_dict={x : [image81]}))
# print('Neural Network predicted', classification[0])
# classification = (sess.run(tf.argmax(y_model, 1), feed_dict={x : [image91]}))
# print('Neural Network predicted', classification[0])


with open('C:/Users/dbstn/Desktop/B.txt', 'w') as f:
  for a in range(10):
    b_out = sess.run(b[a]) #b_out 은 -0.014648308 과 같은것
    #포맷을 바꾸자
    # b_out = format(b_out, "7.6f")
    b_out = float((b_out*256)+0.5)
    b_out = int(b_out)
    b_out = (str(b_out)+', ')
    f.write(str(b_out))
    count = count + 1
    print("B추출을 ", count, "/10 번 완료했습니다.")

wo1 = []
wo1 = [0.000000]*120000 #크기 할당

with open('C:/Users/dbstn/Desktop/W.txt', 'w') as f:
    for a in range(784):
        for b in range(10):
            w_out = sess.run(W[a][b])
            # w_out = format(w_out, "7.6f")
            w_out = float((w_out*256)+0.5)
            w_out = int(w_out)
            w_out = (str(w_out)+', ')
            wo1[(count1%10)*784+(count1//10)]=w_out
            count1 = count1 +1
            print("W추출을 ", count1, "/7840 번 완료했습니다.")
    for c in range(7480):
        f.write(str(wo1[c]))
        print("W입력을 ", count1, "/7840 번 완료했습니다.")

sess.close()








##########################################################   CNN   #############################################################
#
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import random
# from tensorflow.examples.tutorials.mnist import input_data
#
# X = tf.placeholder(tf.float32, [None, 784])  # input, 784개의 값을 가지며 n개의 이미지이다.
# X_img = tf.reshape(X, [-1, 28, 28, 1])  # input 을 이미지로 인식하기 위해 reshape을 해준다. 28*28의 이미지이며 단일색상, 개수는 n개이므로 -1
# Y = tf.placeholder(tf.float32, [None, 10])  # output
#
# ##########################기존 visual의 코드와 동일하게 레이어 구성##############################
# # layer 1
# W1 = tf.Variable(tf.random_normal([5, 5, 1, 20], stddev=0.1))  # 3*3크기의 필터, 색상은 단일, 총 32개의 필터
# L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='VALID')
# L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')  # 2*2의 크기를 2칸씩움직임 max pooling
#
# # layer 2
# W2 = tf.Variable(tf.random_normal([5, 5, 20, 50], stddev=0.1))
# L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='VALID')
# L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
#
# # layer 3
# W3 = tf.Variable(tf.random_normal([4, 4, 50, 500], stddev=0.1))
# L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='VALID')
# L3 = tf.nn.relu(L3)
#
# #layer 4 사실상 reshape
# W4 = tf.Variable(tf.random_normal([1, 1, 500, 10], stddev=0.1))
# L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='VALID')
#
# L4 = tf.reshape(L4, [-1, 1 * 1 * 10])
#
# W5 = tf.get_variable("W5", shape=[1 * 1 * 10, 10], initializer=tf.contrib.layers.xavier_initializer())
#
# b = tf.Variable(tf.random_normal([10]))
# hypothesis = tf.matmul(L4, W5) + b
#
#
#
# ###########################원래 layer 구성#########################
# # # layer 1
# # W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.1))  # 3*3크기의 필터, 색상은 단일, 총 32개의 필터
# # L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')  # conv2d 를 통과해도 28*28 크기를 가짐, 대신 32개의 필터이므로 총 32개의 결과가 생김
# # L1 = tf.nn.relu(L1)
# # L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # max pooling을 하고 나면 스트라이드 및 패딩 설정에 의해 14*14크기의 결과가 나옴
# #
# # # layer 2
# # W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.1))
# # L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
# # L2 = tf.nn.relu(L2)
# # L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# # L2 = tf.reshape(L2, [-1, 7 * 7 * 64])
# #
# # # fully-connected layer
# # W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 10], initializer=tf.contrib.layers.xavier_initializer())
# #
# # b = tf.Variable(tf.random_normal([10]))
# # hypothesis = tf.matmul(L2, W3) + b
#
# count1 = 0
# count2 = 0
# count3 = 0
#
# wo1 = []
# wo1 = [0]*3750 #메모리할당
# wo2 = []
# wo2 = [0]*75000
# wo3 = []
# wo3 = [0]*48000
#
# ###############################출력 test##################################
# # for i in range(500):
# #     wo1[5 * 5 * 1 * (count1 % 20) + (count1 // 20)] = count1
# #     count1 = count1+1
# # for i in range(500):
# #     print(wo1[i])
# # for i in range(9000):
# #     wo2[((count2 % 50) * 180) + (count2 // 1000) + ((count2 // 50) % 20) * 9] = count2
# #     count2 = count2+1
# # for i in range(9000):
# #     print(wo2[i])
# ######################필터수를 줄인 레이어구성################################
# # layer 1
# W1 = tf.Variable(tf.random_normal([5, 5, 1, 10], stddev=0.1))
# L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='VALID')
# L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
#
# # layer 2
# W2 = tf.Variable(tf.random_normal([5, 5, 10, 20], stddev=0.1))
# L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='VALID')
# #L2 = tf.nn.relu(L2)
# L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
#
# #flatten
# L2 = tf.reshape(L2, [-1, 4 * 4 * 20])
#
# #fully connected layer
# W3 = tf.get_variable("W3", shape=[4 * 4 * 20, 10], initializer=tf.contrib.layers.xavier_initializer())
#
# b = tf.Variable(tf.random_normal([10]))
# hypothesis = tf.matmul(L2, W3) + b
#
#
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
# optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
#
# # init
# # sess = tf.Session()
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# training_epochs = 15
# batch_size = 100
#
# # training
# print('Learning started. It takes sometimes.')
# for epoch in range(training_epochs): #15번 학습
#   avg_cost = 0
#   total_batch = int(mnist.train.num_examples / batch_size)
#   #mnist.train.num_examples = 55000이고 batch_size=100이니 batch의 갯수는 550개 = 1iteration
#   # 1iteration은 550개 이고 한개 batch 학습당 100개 데이터를 학습
#   for i in range(total_batch): # 550. 즉 iteration을 도는것 이 안에서는 100개씩 학습한다.
#     batch_xs, batch_ys = mnist.train.next_batch(batch_size) #batch size만큼 읽는다.
#     feed_dict = {X: batch_xs, Y: batch_ys}
#     c, _, = sess.run([cost, optimizer], feed_dict=feed_dict)
#     avg_cost += c / total_batch
#   print("Epoch:", "%04d" % (epoch + 1), "cost =", "{:.9f}".format(avg_cost))
# print('Learning Finished!')
#
# # Test
# correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.arg_max(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
#
# with open('C:/Users/dbstn/Desktop/bias.txt', 'w') as f:
#   for a in range(10):
#     b_out = sess.run(b[a])
#     b_out = (str(b_out) + ', ')
#     f.write(str(b_out))
#     print ( "bias추출을 ", a+1 , "번 완료했습니다.")
#
# with open('C:/Users/dbstn/Desktop/W1.txt', 'w') as f:
#     for a in range(5):
#         for b in range(5):
#             for c in range(1):
#                 for d in range(10):
#                     w_out1 = sess.run(W1[a][b][c][d])
#                     w_out1 = (str(w_out1)+', ')
#                     wo1[5*5*1*(count1%10)+(count1//10)] = w_out1
#                     count1 = count1 +1
#         print("W1추출을 ", count1, "/250 번 완료했습니다.")
#     for e in range(250):
#         f.write(str(wo1[e]))
#
# with open('C:/Users/dbstn/Desktop/W2.txt', 'w') as f:
#     for a in range(5):
#         for b in range(5):
#             for c in range(10):
#                 for d in range(20):
#                     w_out2 = sess.run(W2[a][b][c][d])
#                     w_out2 = (str(w_out2)+', ')
#                     wo2[((count2%20)*250)+(count2//200)+((count2//20)%10)*25] = w_out2
#                     count2 = count2 +1
#                 print("W2추출을 ", count2, "/5000 번 완료했습니다.")
#     for e in range(5000):
#         f.write(str(wo2[e]))
#
# with open('C:/Users/dbstn/Desktop/W3.txt', 'w') as f: #matmul을 위한
#     for a in range(320):
#         for b in range(10):
#             w_out3 = sess.run(W3[a][b])
#             w_out3 = (str(w_out3)+', ')
#             wo3[(count3%10)*320+(count3//10)]=w_out3
#             count3 = count3 +1
#         print("W3추출을 ", count3, "/3200 번 완료했습니다.")
#     for c in range (3200):
#         f.write(str(wo3[c]))
#
# print("finished")