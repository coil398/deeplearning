import tensorflow as tf
import read_data
import image


FLAGS = tf.app.flags.FLAGS
IMAGE_SIZE = read_data.IMAGE_SIZE
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * 3
NUM_CLASSES = read_data.NUM_CLASSES

tf.app.flags.DEFINE_string('train_dir', '/tmp/image_train', 'directory')
tf.app.flags.DEFINE_integer('max_steps', 1000000, 'steps')
tf.app.flags.DEFINE_boolean('log_device_placement',
                            False, 'whether to log device placement')
tf.app.flags.DEFINE_float('learning_rate', 1e-4, 'learning_rate')


def train():
    with tf.Graph().as_default():
        images_placeholder = tf.placeholder(
            'float', shape=(None, IMAGE_PIXELS))
        labels_placeholder = tf.placeholder('float', shape=(None, NUM_CLASSES))

        train_images, train_labels = read_data.get_images(['male', 'female'])

        test_images, test_labels = read_data.get_images(
            ['maleTest', 'femaleTest'])

        logits = image.inference(images_placeholder)

        loss_value = image.loss(logits, labels_placeholder)

        train_op = image.train(loss_value, FLAGS.learning_rate)

        acc = image.accuracy(logits, labels_placeholder)

        saver = tf.train.Saver()

        sess = tf.Session()

        sess.run(tf.initialize_all_variables())

        summary_op = tf.merge_all_summaries()

        summary_writer = tf.train.SummaryWriter(
            FLAGS.train_dir, sess.graph_def)

        for step in range(FLAGS.max_steps):
            for i in range(int(len(train_images) / FLAGS.batch_size)):
                batch = FLAGS.batch_size * i
                sess.run(train_op, feed_dict={
                    images_placeholder: train_images[batch:batch + FLAGS.batch_size],
                    labels_placeholder: train_labels[batch:batch + FLAGS.batch_size],
                })

            train_accuracy = sess.run(acc, feed_dict={
                images_placeholder: train_images,
                labels_placeholder: train_labels,
            })
            print('step %d, training accuracy %g' % (step, train_accuracy))

            summary_str = sess.run(summary_op, feed_dict={
                images_placeholder: train_images,
                labels_placeholder: train_labels,
            })
            summary_writer.add_summary(summary_str, step)

    print('test accuracy %g' % sess.run(acc, feed_dict={
        images_placeholder: test_images,
        labels_placeholder: test_labels,
    }))

    saver.save(sess, 'model.ckpt')


def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()
