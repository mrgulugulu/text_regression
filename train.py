import tensorflow as tf
import numpy as np
import os
import datetime
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import data_helpers
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)



def get_flags():
  """
  desc: get cli arguments
  returns:
    args: dictionary of cli arguments
  """

  tf.flags.DEFINE_string("train_path", "data_training1.xlsx", "path")#模型训练文本输入的地址
  tf.flags.DEFINE_string("run_type", "train",
                         "enter train or test to specify run_type (default: train)")
  tf.flags.DEFINE_integer("embedding_dim", 100,
                          "Dimensionality of character embedding (default: 100)")
  tf.flags.DEFINE_float("dev_sample_percentage", 0.1, "dev")
  tf.flags.DEFINE_integer("max_sentence_length", 120, "l")
  tf.flags.DEFINE_integer("max_paragraph_length", 120, "l")#改动这里的话要一同改动下面的截取
  tf.flags.DEFINE_bool("allow_soft_placement", True, "1")
  tf.flags.DEFINE_bool("log_device_placement", False, "1")
  tf.flags.DEFINE_bool("gpu_allow_growth", True, "1")
  tf.flags.DEFINE_integer("class_num", 1, "l")
  tf.flags.DEFINE_integer("display_every", 10, "1")

  tf.flags.DEFINE_integer("batch_size", 32,
                          "Batch Size (default: 2)")
  tf.flags.DEFINE_integer("num_epochs", 25,
                          "Number of training epochs (default: 25)")
  tf.flags.DEFINE_integer("evaluate_every", 500,
                          "Evaluate model on dev set after this many steps")
  tf.flags.DEFINE_integer("log_summaries_every", 30,
                          "Save model summaries after this many steps (default: 30)")
  tf.flags.DEFINE_integer("checkpoint_every", 100,
                          "Save model after this many steps (default: 100)")
  tf.flags.DEFINE_integer("num_checkpoints", 5,
                          "Number of checkpoints to store (default: 5)")
  tf.flags.DEFINE_float("max_grad_norm", 5.0,
                        "maximum permissible norm of the gradient (default: 5.0)")
  tf.flags.DEFINE_float("dropout_keep_proba", 0.5,
                        "probability of neurons turned off (default: 0.5)")
  tf.flags.DEFINE_float("learning_rate", 0.00001,
                        "model learning rate (default: 0.001)")
  tf.flags.DEFINE_float("per_process_gpu_memory_fraction", 0.95,
                        "gpu memory to be used (default: 0.90)")

  FLAGS = tf.flags.FLAGS
  FLAGS.flag_values_dict()

  return FLAGS


def train():
    FLAGS = get_flags()
    x_text, y = data_helpers.load_data_and_labels(FLAGS.train_path) #y应该是一个batchsize*1的矩阵，也就是tfidf的矩阵
    x_text = np.array(x_text)
    np.random.seed(12)
    #shuffle_indices = np.random.permutation(np.arange(len(y)))  # len(矩阵)返回行数
    #x_text = x_text[shuffle_indices]    #打乱顺序

    y = np.array(y).reshape((len(y), 1))
    #y_shuffled = y[shuffle_indices]

    y_shuffled = y
    x_text = list(x_text)
    temp = sum(x_text,[])   #列表降维
    #temp = list(np.array(x_text).reshape(-1))

    dev_sample_index = -1 * int(0.9 * float(len(y)))  # -1使得x_train截至倒数1/10处，即长度为0.9
    x_text_train_list, x_text_dev_list = x_text[:dev_sample_index], x_text[dev_sample_index:]  # 将训练集分割为9：1
    y_train1, y_dev1 = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]# y_shuffled要reshape

    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
    _ = np.array(list(vocab_processor.fit_transform(temp)))    #建立所有文档字的id映射表
    #输入文本，建立词汇id映射表，将句子单词id化，与上句一起用
    count = -1
    x_train = []
    x_dev = []
    i = 1

    for paragraph in x_text_train_list:
        x_document = np.array(list(vocab_processor.fit_transform(paragraph)))
        x = np.array(x_document)
        x_len = len(x)
        i += 1
        if x_len < FLAGS.max_paragraph_length:
            #for i in range(FLAGS.max_paragraph_length - x_len):
            try:
                x = np.pad(x, ((0, FLAGS.max_paragraph_length - x_len), (0, 0)))
            except ValueError:
                print('something wrong in training_data NO.{}'.format(i))
                pass
        else:
            x = x[:150]#这里要与"max_paragraph_length"一起改


        x_train.append(x)
    x_train = np.array(x_train)
    y_train = np.array(y_train1)
    batches = zip(x_train, y_train)

    for paragraph in x_text_dev_list:
        x_document = np.array(list(vocab_processor.fit_transform(paragraph)))
        x = np.array(x_document)
        x_len = len(x)
        i += 1
        if x_len < FLAGS.max_paragraph_length:
            # for i in range(FLAGS.max_paragraph_length - x_len):
            try:
                x = np.pad(x, ((0, FLAGS.max_paragraph_length - x_len), (0, 0)))
            except ValueError:
                print('something wrong in dev_data NO.{}'.format(i))
                pass
        else:
            x = x[:150]

        x_dev.append(x)
    x_dev = np.array(x_dev)
    y_dev = np.array(y_dev1)


    with tf.Graph().as_default():       # 实例化一个类，用于tensorflow计算和表示的数据流图；as_default表示将该图作为整个tensorflow运行的默认图
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.per_process_gpu_memory_fraction)
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,        # 如果你指定的设备不存在，允许TF自动分配设备
            log_device_placement=FLAGS.log_device_placement,gpu_options=gpu_options)        # 是否打印设备分配日志
        session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth      # 使得GPU内存不会一开始就占满
        session_conf.gpu_options.allocator_type = "BFC"
        sess = tf.Session(config=session_conf)
        with sess.as_default():                                         # 设为默认
            '''model = AttLSTM(
                sequence_length=FLAGS.max_sentence_length,#x_train.shape[1],       #shape[1]为列数，即句子长度
                num_classes=FLAGS.class_num,#y_train.shape[1],           #shape[1]为关系数目
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                hidden_size=FLAGS.hidden_size,
                l2_reg_lambda=FLAGS.l2_reg_lambda)'''
            model = HAN(
                      num_classes=1,
                      vocab_size=len(vocab_processor.vocabulary_),
                      embedding_size=FLAGS.embedding_dim,
                      max_grad_norm=FLAGS.max_grad_norm,
                      #dropout_keep_proba=FLAGS.dropout_keep_proba,
                      learning_rate=FLAGS.learning_rate)

            # Define Training procedure
            '''global_step = tf.Variable(0, name="global_step", trainable=False)                       # .minimize(loss，global_step),可分为compute_gradients函数和apply_gradients；会自动使该全局步数加一；滑动平均模型和学习率的指数衰减模型中
            optimizer = tf.train.AdadeltaOptimizer(FLAGS.learning_rate, FLAGS.decay_rate, 1e-6)     # decay_rate 为了防止学习率过大，在收敛到全局最优点的时候会来回摆荡，所以要让学习率随着训练轮数不断按指数级下降，收敛梯度下降的学习步长。
            gvs = optimizer.compute_gradients(model.loss)                                           # 计算loss中可训练的var_list中的梯度。相当于minimize()的第一步，返回(gradient, variable)对的list。
            capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]            # tf.clip_by_value(A, min, max)：输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。
            train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)               # #执行对应变量的更新梯度操作
'''
            global_step = tf.Variable(0, name="global_step", trainable=False)
            tvars = tf.trainable_variables()
            grads, global_norm = tf.clip_by_global_norm(tf.gradients(model.loss, tvars),
                                                        model.max_grad_norm)
            optimizer = tf.train.AdamOptimizer(model.learning_rate)
            train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                 name="train_op",
                                                 global_step=global_step)
            # Output directory for models and summaries
            timestamp = str(int(time.time()))                                                   # time.time() 1970年至今的秒数
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))          #curdir指代当前目录
            print("Writing to {}\n".format(out_dir))                                            #该目录存储模型等

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", model.loss)            #tf.summary.scalar 用来显示标量信息，一般在画loss,accuary时会用到这个函数,用tensorboard可以看
            acc_summary = tf.summary.scalar("accuracy", model.accuracy)

            # Train Summaries 训练集
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries 测试集
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)    #实例化Saver

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Generate batches
            #batches = data_helpers.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)  #x_train：（8000*0.9，90）；y_train: (8000*0.9,19)

            # Training loop. For each batch...
            #best_f1 = 0.0  # For save checkpoint(model)
            for i_ in range(100):
                #batches = zip(x_train, y_train)
                batches = data_helpers.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
                for batch in batches:

                    x_batch, y_batch = zip(*batch)      #此时x，y获得的都是id化的句子了 x_train.shape = (batch_size,90), y_train.shape = (batch_size,19)
                    # Train
                    feed_dict = {
                        model.input_x: x_batch,
                        model.input_y: y_batch,
                        model.is_training: True,
                        model.dropout_keep_proba: FLAGS.dropout_keep_proba
                    }
                    _, step, summaries, loss, prediction = sess.run(
                        [train_op, global_step, train_summary_op, model.loss, model.logits], feed_dict)
                    train_summary_writer.add_summary(summaries, step)

                    # Training log display
                    if step % FLAGS.display_every == 0:     #display_every==10，即每10步
                        time_str = datetime.datetime.now().isoformat()
                        print("{}: step {}, loss {:g}".format(time_str, step, loss))    #eg.  2019-10-16T19:58:12.908340: step 420, loss 3.71331, acc 0.8

                    # Evaluation
                    if step % FLAGS.evaluate_every == 0:    #evaluate_every==100
                        print("\nEvaluation:")
                        predictions = []
                        accuracys = []
                        batches_dev = data_helpers.batch_iter(list(zip(x_dev, y_dev)), FLAGS.batch_size,
                                                              FLAGS.num_epochs)
                        for batch in batches_dev:
                            x_dev_data, y_dev_data = zip(*batch)
                            feed_dict = {
                                model.input_x: x_dev_data,
                                model.input_y: y_dev_data,
                                model.is_training: True,
                                #model.emb_dropout_keep_prob: 1.0,
                                #model.rnn_dropout_keep_prob: 1.0,
                                model.dropout_keep_proba: 1.0
                            }
                            summaries, loss, accuracy, prediction = sess.run(
                                [dev_summary_op, model.loss, model.accuracy, model.logits], feed_dict)

                            for p in prediction:
                                predictions.append(list(p))#这里应该算出来一个值

                            #accuracys.append(accuracy)
                            dev_summary_writer.add_summary(summaries, step)

                        predictions_array = np.array(predictions)
                        predictions_array2 = predictions_array
                        time_str = datetime.datetime.now().isoformat()
                        y_truth = np.array(y_dev)
                        print(y_truth.shape,predictions_array2.shape)


                        print("{}: step {}, loss {:g}".format(time_str, step, loss))

                        # Model checkpoint
                        if step % 4000 == 0:#每4000步保存一次
                            '''best_f1 = f1'''
                            path = saver.save(sess, checkpoint_prefix + "-{:.3g}".format(loss), global_step=step)    # "-{:.3g}".format(best_f1)表示将F1存入模型名字；global_step=step表示将步数存入名字
                            print("Saved model checkpoint to {}\n".format(path))



def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()
