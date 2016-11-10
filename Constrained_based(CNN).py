# borrowed form "https://github.com/dennybritz/cnn-text-classification-tf" with modifications
import operator
import logging
import os
import time
import datetime

import numpy as np
import tensorflow as tf

import data_preprocess
from text_cnn import TextCNN



# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 3, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS.batch_size
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.iteritems()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparatopn
# ==================================================

# Load train and validation data
print("Loading data...")

x, y, vocabulary_inv, labels = data_preprocess.load_data()

x_tr , x_dev = x[:], x[:]
y_tr , y_dev = y[:], y[:]
vocabulary_all = {x: i for i, x in enumerate(vocabulary_inv)}

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y_tr)))
x_shuffled = x_tr[shuffle_indices]
y_shuffled = y_tr[shuffle_indices]


x_train = x_shuffled[:]
y_train = y_shuffled[:]

print("Vocabulary Size: {:d}".format(len(vocabulary_all)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
print 'shape: ', x_train.shape



logging.basicConfig(filename='result_LR_SQ.log', level=logging.INFO)

def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def predc_vec2prop(pred_vec):
    predicted_vector = pred_vec.tolist()  # [.32,.28,.74,...]
    predicted_props_id = {predicted_vector.index(ind): ind for ind in predicted_vector}  # {1:.32,2:.28,3:.74,...}
    predicted_props_id_sorted = sorted(predicted_props_id.items(), key=operator.itemgetter(1), reverse=True)
    predicted_props_id_value = [(labels[pr[0]], pr[1]) for pr in predicted_props_id_sorted]
    return predicted_props_id_value

answers={}
with tf.Graph().as_default():

    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=len(labels),
            vocab_size=len(vocabulary_all),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=map(int, FLAGS.filter_sizes.split(",")),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "models", timestamp))
        print 'num_class', len(labels)
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "models")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            if step % 100 == 0:

                print("trn: {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy, predicted_array = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.predicted_array], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            # if step % 1000 == 0:

            print("dev: {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
            return accuracy, predicted_array

        # Generate batches
        def candidate_answer(step):  # train of test
            if step == 'train':
                batches_train = batch_iter(zip(x_train, y_train), FLAGS.batch_size, FLAGS.num_epochs)
                # Training loop. For each batch...


                for batch in batches_train:

                    x_batch, y_batch = zip(*batch)
                    train_step(x_batch, y_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % FLAGS.evaluate_every == 0:
                        val_accuracy_current, _ = dev_step(x_dev, y_dev, writer=dev_summary_writer)


                    if current_step % FLAGS.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        #path = saver.save(sess, '/hawork/people/ebrahimian/WDS/SQ_65.16_pulished/models/model')

                        print("Saved model checkpoint to {}\n".format(path))
            elif step == 'test':
                model ='/ha/home/ebrahimian/Desktop/mwe/tag_classifier/models/1477928529/checkpoints/models-500'

                saver.restore(sess,model)
                test_accuracy, predicted_array = dev_step(x_dev, y_dev, writer=dev_summary_writer)


                print 'test', test_accuracy

if __name__ == '__main__':
    t1 = time.time()
    candidate_answer('train')

    t2 = time.time()
    print "it took %d minues" % ((t2 - t1) / 60)
