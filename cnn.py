#
# import tensorflow as tf
#
# #---------------------------------------------------------------------------------------------------- 1. MNIST 데이터를 가져옵니다.
# # MNIST 데이터 관련 내용은 다음 포스팅 참고
# #
# # Tensorflow 예제 - MNIST 데이터 출력해보기 ( http://webnautes.tistory.com/1232 )
# from tensorflow.examples.tutorials.mnist import input_data
#
# # one_hot을 True로 설정하면 라벨을 one hot 벡터로 합니다. False이면 정수형 숫자가 라벨이 됩니다.
# # /tmp/data/ 폴더를 생성하고 MNIST 데이터 압축파일을 다운로드 받아 압축을 풀고 데이터를 읽어옵니다.
# # 이후에는 다운로드 되어있는 압축파일의 압축을 풀어서 데이터를 읽어옵니다.
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
#
# #---------------------------------------------------------------------------------------------------- 2. 모델을 생성합니다.
# # 모델의 입력을 받기 위해 플레이스홀더를 사용합니다.
# # 첫번째 차원이 None인 이유는 데이터 개수 제약없이 입력 받기 위해서입니다.
# # 두번째 차원이 784인 것은  MNIST의 이미지의 크기가 28 x 28 = 784 픽셀이기 때문입니다.
# x = tf.placeholder(tf.float32, [None, 784])
#
# # 모델 파라미터
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
#
# # softmax를 사용한 모델을 생성
# # W, b 모델 파라미터 -> 변수
# # x 이미지 데이터 입력 -> 플레이스홀더
# y_model = tf.matmul(x, W) + b #행렬 곱연산과 bias 더해주기
#
#
# #---------------------------------------------------------------------------------------------------- 3. loss와 optimizer를 정의합니다.
# y = tf.placeholder(tf.float32, [None, 10])  # 크기 10인 MNIST의 라벨 데이터 (숫자가 열개니깐)
#
# # 크로스 엔트로피(cross entropy) 함수 공식을 그대로 사용하면 수치적으로 불안정하여 계산 오류가 발생할 수 있습니다.
# # cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(tf.nn.softmax(y_model)), reduction_indices=1))
# #
# # 그래서 tf.nn.softmax_cross_entropy_with_logits_v2를 사용합니다. (tf.nn.softmax_cross_entropy_with_logits는 deprecated 되었습니다.)
#
# # cross entropy를 손실 함수(cost function)로 사용
# #cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(tf.nn.softmax(y_model)), reduction_indices=1))
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_model))
#
# # Gradient Descent - Backpropagation 기법으로 최적화
# optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost) # learning_rate = 0.01
#
#
#
# #---------------------------------------------------------------------------------------------------- 4. 훈련을 위한 세션 시작
# sess = tf.Session()
#
#
# sess.run(tf.global_variables_initializer()) # 변수 초기화
#
#
# for epoch in range(25): # 훈련을 25번 반복
#     avg_cost = 0.
#
#     # 1번 훈련시 전체 훈련 데이터를 사용하려면 100개씩 몇번 가져와야 하는지 계산하여 반복
#     total_batch = int(mnist.train.num_examples / 100)
#     for i in range(total_batch):
#         # 전체 훈련 데이터(mnist.train)에서 100개씩 데이터를 가져옵니다.
#         # (100, 784) (100, 10)
#         batch_xs, batch_ys = mnist.train.next_batch(100)
#
#         # optimizer와 cost 오퍼레이션을 실행합니다.
#         _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
#
#         # 현재까지 평균 손실(loss)를 누적합니다.
#         avg_cost += c / total_batch
#
#     # 훈련 1번 끝날때 마다 중간 결과를 출력
#     print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
#
#
# print("최적화 완료")
#
# # with open('C:/Users/dbstn/Desktop/weight_data.txt', 'w') as f:
# #     for i in range(784):
# #         for j in range(10):
# #             w_out = sess.run(W[i][j])
# #             w_out = round(w_out, 4)
# #             f.write(str(w_out)+',')
# #         print(i)
#
# #---------------------------------------------------------------------------------------------------- 5. 정확도 측정
# # 라벨값 y와 모델로 계산된 값 y_model이 똑같이 같은 인덱스가 제일 크다고 하는지 검사
# # ( tf.argmax 함수가 배열에서 가장 큰 값을 가리키는 인덱스를 리턴합니다.. )
# # 결과적으로 correct_prediction는 True 또는 False 값의 리스트를 가지게 됨
# #
# # tf.argmax에 대한 자세한 내용은 다음 포스팅 참고
# # Tensorflow 예제 - tf.argmax 함수 사용법 ( http://webnautes.tistory.com/1234 )
# correct_prediction = tf.equal(tf.argmax(y_model, 1), tf.argmax(y, 1))
#
#
# # tf.cast 함수를 사용하여 True 또는 False를 실수 1 또는 0으로 변환
# # 전체 데이터가 일치한다면 모든 값이 1이며 평균인 accuracy는 1이 되어야 합니다.
# #
# # tf.reduce_mean에 대한 자세한 내용은 다음 포스팅 참고
# # Tensorflow 예제 - tf.reduce_mean 함수 사용법 ( http://webnautes.tistory.com/1235 )
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# # 정확도 측정을 위해서 훈련 데이터(mnist.train) 대신에 별도의 테스트 데이터(mnist.test)를 사용해야 합니다.
# print("Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
#
# sess.close()




# import tensorflow as tf
# import numpy as np
#
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#
# x_val  = x_train[50000:60000] #validation data 는 10000개 이다.
# x_train = x_train[0:50000] #train data는 50000개 이다.
# y_val  = y_train[50000:60000]
# y_train = y_train[0:50000]
#
# print("train data has " + str(x_train.shape[0]) + " samples")
# print("every train data is " + str(x_train.shape[1]) + " * " + str(x_train.shape[2]) + " image")
# print("validation data has " + str(x_val.shape[0]) + " samples")
# print("every train data is " + str(x_val.shape[1])+ " * " + str(x_train.shape[2]) + " image")
# # sample to show gray scale values
# print(x_train[0][8])
# # sample to show labels for first train data to 10th train data
# print(y_train[0:9])
# print("test data has " + str(x_test.shape[0]) + " samples")
# print("every test data is " + str(x_test.shape[1]) + " * " + str(x_test.shape[2]) + " image")
#
#
# x_train = np.reshape(x_train, (50000,28,28,1))
# x_val = np.reshape(x_val, (10000,28,28,1))
# x_test = np.reshape(x_test, (10000,28,28,1))
#
# print(x_train.shape)
# print(x_test.shape)
# x_train = x_train.astype('float32')
# x_val = x_val.astype('float32')
# x_test = x_test.astype('float32')
#
# gray_scale = 255
# x_train /= gray_scale
# x_val /= gray_scale
# x_test /= gray_scale
#
# num_classes = 10
# y_train = tf.keras.utils.to_categorical(y_train, num_classes)
# y_val = tf.keras.utils.to_categorical(y_val, num_classes)
# y_test = tf.keras.utils.to_categorical(y_test, num_classes)
#
# x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
# y_ = tf.placeholder(tf.float32, shape=[None, 10])
#
# def weight_variable(shape):
#   initial = tf.truncated_normal(shape, stddev=0.1)
#   return tf.Variable(initial)
#
# def bias_variable(shape):
#   initial = tf.constant(0.1, shape=shape)
#   return tf.Variable(initial)
#
# def conv2d(x, W):
#   return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#
# def max_pool_2x2(x):
#   return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1], padding='SAME')
#
#
# W_conv1 = weight_variable([5, 5, 1, 16])
# b_conv1 = bias_variable([16])
# h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1) #conv와 bias를 합친다.
# h_pool1 = max_pool_2x2(h_conv1) #max pooling
#
# W_conv2 = weight_variable([5, 5, 16, 32])
# b_conv2 = bias_variable([32])
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)
#
# W_fc1 = weight_variable([7 * 7 * 32, 128])
# b_fc1 = bias_variable([128])
#
# h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*32])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
# W_fc2 = weight_variable([128, 10])
# b_fc2 = bias_variable([10])
#
# y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
#
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv)) #y_conv가 돌아온값 y_는 맞는값. 비교해준다.
# train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
#
# correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) #argmax : 가장 큰 인덱스를 리턴 즉 같다면 correct_prediction이 1
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #전체 평균을 구한다. 모든 차원이 제거되고 하나의 스칼라값이 출력 cast: casting연산
#
# # initialize
# init = tf.global_variables_initializer()
#
# # train hyperparameters
# epoch_cnt = 3
# #batch_size = 10
# batch_size = 500
# #iteration = 10
# iteration = len(x_train) // batch_size # 나누기 이후 소수점 이하의 수를 버리고 정수 부분의 수만 구함
#
# # Start training
# with tf.Session() as sess:
#     tf.set_random_seed(777)
#     # Run the initializer
#     sess.run(init)
#     for epoch in range(epoch_cnt):
#         avg_loss = 0.
#         start = 0;
#         end = batch_size
#
#         for i in range(iteration):
#             if i % 10 == 0:
#             #if i%1==0:
#                 train_accuracy = accuracy.eval(feed_dict={x: x_train[start: end], y_: y_train[start: end]})
#                 print("step " + str(i) + ": training accuracy: " + str(train_accuracy))
#             train_step.run(feed_dict={x: x_train[start: end], y_: y_train[start: end]})
#             start += batch_size;
#             end += batch_size
#
#             # Validate model
#         val_accuracy = accuracy.eval(feed_dict={x: x_val, y_: y_val})
#         print("validation accuracy: " + str(val_accuracy))
#
#     test_accuracy = accuracy.eval(feed_dict={x: x_test, y_: y_test})
#     print("test accuracy: " + str(test_accuracy))
#
#     # with open('C:/Users/dbstn/Desktop/weight_data.txt', 'w') as f:
#     #     for i in range(784):
#     #         w_out = sess.run(b_conv1[i])
#     #         f.write(str(w_out)+'\n')







# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# import tensorflow as tf
#
#
# x = tf.placeholder('float', [None, 784])
# x_image = tf.reshape(x, [-1, 28, 28, 1])
#
#
# def weight_variable(shape):
#     # truncate values whose magnitude is more than 2 standard deviations
#     initial = tf.truncated_normal(shape, mean=0.0, stddev=0.1)
#     return tf.Variable(initial)
#
# def bias_variable(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)
#
#
# def conv2d(x, W):
#     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#
# def max_pool_2x2(x):
#     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#
#
# W_conv1 = weight_variable([5, 5, 1, 32])
# b_conv1 = bias_variable([32])
# z_conv1 = conv2d(x_image, W_conv1) + b_conv1
#
# h_conv1 = tf.nn.relu(z_conv1)
# h_pool1 = max_pool_2x2(h_conv1)
#
# print('x', x_image.get_shape())
# print('W', W_conv1.get_shape())
# print('b', b_conv1.get_shape())
#
# print('h_conv1', h_conv1.get_shape())
# print('h_pool1', h_pool1.get_shape())
#
# W_conv2 = weight_variable([5, 5, 32, 64])
# b_conv2 = bias_variable([64])
#
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)
#
# print('h_conv2', h_conv2.get_shape())
# print('h_pool2', h_pool2.get_shape())
#
# h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64]) #flat하는 과정 필요
#
# W_fc1 = weight_variable([7 * 7 * 64, 1024])
# b_fc1 = bias_variable([1024])
#
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
#
# keep_prob = tf.placeholder('float')
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# W_fc2 = weight_variable([1024, 10])
# b_fc2 = bias_variable([10])
#
# h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#
# y = tf.nn.softmax(h_fc2)
#
# y_label = tf.placeholder('float', [None, 10])
# cross_entropy = -tf.reduce_sum(y_label * tf.log(y))
# train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_label, 1)) # 값이 같으면 correct_prediction =1
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
#
#
# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
#
# for i in range(20000):
#     batch_x, batch_y = mnist.train.next_batch(50)
#
#     if i % 1000 == 0:
#         train_accuracy = sess.run(accuracy, feed_dict={
#             x: batch_x,
#             y_label: batch_y,
#             keep_prob: 1.0
#         })
#         print('step %d, training accuracy %g' % (i, train_accuracy))
#
#     sess.run(train, feed_dict={
#         x: batch_x,
#         y_label: batch_y,
#         keep_prob: 0.5
#     })
#
# test_accuracy = sess.run(accuracy, feed_dict={
#         x: mnist.test.images,
#         y_label: mnist.test.labels,
#         keep_prob: 1.0
#     })
# print('test accuracy %g' % test_accuracy)
#






# from tensorflow.examples.tutorials.mnist import input_data
# import tensorflow as tf
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# sess = tf.InteractiveSession()
#
# #입력될 이미지와 각각의 출력 클래스에 해당하는 노드를 생성하는 것. = 새로운 placeholder를 추가
# x = tf.placeholder(tf.float32, shape=[None, 784])
# y_ = tf.placeholder(tf.float32, shape=[None, 10])
#
# #가중치 W와 편향b를 정의한다.
# W = tf.Variable(tf.zeros([784,10])) #둘다 0으로 된 텐서로 초기화를 한다. 이제부터 W와 b를 학습시켜나갈것
# b = tf.Variable(tf.zeros([10]))
#
# #Variable들은 세션이 시작되기전에 초기화 되어야 한다.
# sess.run(tf.global_variables_initializer())
#
# #각각의 클래스에 대한 소프트맥스 함수의 결과를 계산할 수 있다.
# y = tf.nn.softmax(tf.matmul(x,W) + b)
#
# #사용된 이미지들 각각에서 계산된 합의 평균을 구하는 함수
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#
# #대칭성을 깨뜨리고 기울기가 0이 되는 것을 방지하기 위해 가중치에 약간의 잡음 (0.1)을 주어 초기화한다.
# def weight_variable(shape):
#   initial = tf.truncated_normal(shape, stddev=0.1)
#   return tf.Variable(initial)
#
# #모델에 ReLU뉴런이 포함되므로 죽은 뉴런을 방지하기 위해 편향을 작은양수 0.1로 초기화해준다.
# def bias_variable(shape):
#   initial = tf.constant(0.1, shape=shape)
#   return tf.Variable(initial)
#
# #스트라이드를 1로(필터가 이동할 간격), 출력크기가 입력과 같게 되도록 0으로 패딩하도록 설정함
# #제로패딩 (1폭) 을하면 출력크기가 작아지지 않는다.
# def conv2d(x, W):
#   return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#
# #2*2 크기의 맥스풀링을 적용했다.
# def max_pool_2x2(x):
#   return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1], padding='SAME')
#
# W_conv1 = weight_variable([5, 5, 1, 32])
# b_conv1 = bias_variable([32])
# x_image = tf.reshape(x, [-1,28,28,1])
# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1)
#
# W_conv2 = weight_variable([5, 5, 32, 64])
# b_conv2 = bias_variable([64])
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)
#
# W_fc1 = weight_variable([7 * 7 * 64, 1024])
# b_fc1 = bias_variable([1024])
#
# h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
# W_fc2 = weight_variable([1024, 10])
# b_fc2 = bias_variable([10])
#
# y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#
# #####모델의 훈련 및 평가#####
#
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
# #학습비율 0.5로 경사하강법을 적용하여 크로스 엔트로피를 최소화하도록 하려면
# #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# #학습을 실행시키기 전 작성한 변수들을 초기화 하는 작업
# sess.run(tf.global_variables_initializer())
#
# #for i in range(20000):
# for i in range(500):
#   batch = mnist.train.next_batch(50) #무작위로 선택된 50개의 데이터로 구성된 batch를 가져온다.
#   if i%100 == 0:
#     train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
#     print("step %d, training accuracy %g"%(i, train_accuracy))
#   #placeholder의 자리에 데이터를 넣을 수 있도록 train_step을 실행하여 배치 데이터를 넘긴다.
#   train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
#
# # with open('C:/Users/dbstn/Desktop/W1.txt', 'w') as f:
# #   for i in range(1024):
# #     w_out = sess.run(W_fc1[i])
# #     f.write(str(w_out))
# #
# # with open('C:/Users/dbstn/Desktop/W2.txt', 'w') as f:
# #   for i in range(1024):
# #     w_out = sess.run(W_fc2[i])
# #     f.write(str(w_out))
# #
# # with open('C:/Users/dbstn/Desktop/B1.txt', 'w') as f:
# #   for i in range(1024):
# #     w_out = sess.run(b_fc1[i])
# #     f.write(str(w_out))
# #
# # with open('C:/Users/dbstn/Desktop/B2.txt', 'w') as f:
# #   for i in range(1024):
# #     w_out = sess.run(b_fc2[i])
# #     f.write(str(w_out))
#
# print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))



import tensorflow as tf
import matplotlib.pyplot as plt
import random
from tensorflow.examples.tutorials.mnist import input_data

X = tf.placeholder(tf.float32, [None, 784])  # input, 784개의 값을 가지며 n개의 이미지이다.
X_img = tf.reshape(X, [-1, 28, 28, 1])  # input 을 이미지로 인식하기 위해 reshape을 해준다. 28*28의 이미지이며 단일색상, 개수는 n개이므로 -1
Y = tf.placeholder(tf.float32, [None, 10])  # output

# layer 1
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.1))  # 3*3크기의 필터, 색상은 단일, 총 32개의 필터
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1],
                  padding='SAME')  # conv2d 를 통과해도 28*28 크기를 가짐, 대신 32개의 필터이므로 총 32개의 결과가 생김
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME')  # max pooling을 하고 나면 스트라이드 및 패딩 설정에 의해 14*14크기의 결과가 나옴

# layer 2
# 이번에는 64개의 필터
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.1))
# conv2d layer를 통과시키면, [?,14,14,64] 형태를 가짐
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
# max pooling 에서 stride가 2 이므로, 결과는 7 * 7 형태를 가질
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# 이후 쭉 펼친다.
L2 = tf.reshape(L2, [-1, 7 * 7 * 64])

# fully-connected layer
W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L2, W3) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# init
# sess = tf.Session()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# training_epochs = 15
training_epochs=1
batch_size = 100

# train
print('Learning started. It takes sometimes.')
for epoch in range(training_epochs):
  avg_cost = 0
  total_batch = int(mnist.train.num_examples / batch_size)
  for i in range(total_batch):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    feed_dict = {X: batch_xs, Y: batch_ys}
    c, _, = sess.run([cost, optimizer], feed_dict=feed_dict)
    avg_cost += c / total_batch
  print("Epoch:", "%04d" % (epoch + 1), "cost =", "{:.9f}".format(avg_cost))
print('Learning Finished!')

# Test
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

with open('C:/Users/dbstn/Desktop/W1.txt', 'w') as f:
  for a in range(3):
    for b in range(3):
      for c in range(1):
        for d in range(32):
          w_out = sess.run(W1[a][b][c][d])
          f.write(str(w_out))