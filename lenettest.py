# 샘플코드와 동일하게 채널을 구성후 텐서 순서대로 뽑은코드
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
# # # layer 1
# # W1 = tf.Variable(tf.random_normal([5, 5, 1, 20], stddev=0.1))  # 3*3크기의 필터, 색상은 단일, 총 32개의 필터
# # L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='VALID')
# # L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')  # 2*2의 크기를 2칸씩움직임 max pooling
# #
# # # layer 2
# # W2 = tf.Variable(tf.random_normal([5, 5, 20, 50], stddev=0.1))
# # L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='VALID')
# # L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
# #
# # # layer 3
# # W3 = tf.Variable(tf.random_normal([4, 4, 50, 500], stddev=0.1))
# # L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='VALID')
# # L3 = tf.nn.relu(L3)
# #
# # #layer 4 사실상 reshape
# # W4 = tf.Variable(tf.random_normal([1, 1, 500, 10], stddev=0.1))
# # L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='VALID')
# #
# # L4 = tf.reshape(L4, [-1, 1 * 1 * 10])
# #
# # W5 = tf.get_variable("W5", shape=[1 * 1 * 10, 10], initializer=tf.contrib.layers.xavier_initializer())
# #
# # b = tf.Variable(tf.random_normal([10]))
# # hypothesis = tf.matmul(L4, W5) + b
# #
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
# training_epochs = 10
# batch_size = 100
#
# # training
# print('Learning started. It takes sometimes.')
# for epoch in range(training_epochs): #10번 학습
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
#     #텐서의 구성은 사이즈 x 사이즈 x 커널
#     for a in range(5): #사이즈
#         for b in range(5): #사이즈
#             for c in range(1): #채널
#                 for d in range(10): #필터
#                     w_out1 = sess.run(W1[a][b][c][d])
#                     w_out1 = (str(w_out1)+', ')
#                     wo1[count1]=w_out1
#                     # wo1[(count1//10)+25*(count1%10)] = w_out1 #채널 필터 사이즈 사이즈 로 뽑기위해
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
#                     wo2[count2]=w_out2
#                     # wo2[(count2//200)+25*(count2%200)] = w_out2
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
#             wo3[count3]=w_out3
#             # wo3[(count3%10)*320+(count3//10)]=w_out3
#             count3 = count3 +1
#         print("W3추출을 ", count3, "/3200 번 완료했습니다.")
#     for c in range (3200):
#         f.write(str(wo3[c]))
#
# print("finished")


#정확도를 높이기 위해 디테일에 대한 고려를 하기 위해 32x32 인풋으로 바꿈

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten

mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test = mnist.test.images, mnist.test.labels

assert(len(X_train) == len(y_train))
assert(len(X_validation) == len(y_validation))
assert(len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))

# Pad images with 0s
X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
X_validation = np.pad(X_validation, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

print("Updated Image Shape: {}".format(X_train[0].shape))

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(1,1))
plt.imshow(image, cmap="gray")
print(y_train[index])

X_train, y_train = shuffle(X_train, y_train)

EPOCHS = 20
BATCH_SIZE = 128

def LeNet(x):
    # Hyperparameters
    mu = 0
    sigma = 0.1
    layer_depth = {
        'layer_1': 6,
        'layer_2': 16,
        'layer_3': 120,
        'layer_f1': 84
    }

    conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    conv1 = tf.nn.relu(conv1)

    pool_1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    conv2 = tf.nn.relu(conv2)

    pool_2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    fc1 = flatten(pool_2)

    fc1_w = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1, fc1_w) + fc1_b

    fc1 = tf.nn.relu(fc1)

    fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_w) + fc2_b

    fc2 = tf.nn.relu(fc2)

    fc3_w = tf.Variable(tf.truncated_normal(shape=(84, 10), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    return logits

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)

rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet')
    print("Model saved")

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
