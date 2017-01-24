import tensorflow as tf
from datetime import datetime
import time
import read_data
import image


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/image_train', 'directory')
tf.app.flags.DEFINE_integer('max_steps', 1000000, 'steps')
tf.app.flags.DEFINE_boolean('log_device_placement',
                            False, 'whether to log device placement')


def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, name='global_step', trainable=False)

        train_images, train_labels = read_data.main(['male', 'female'])

        logits = image.inference(train_images)

        loss = image.loss(logits, train_labels)

        train_op = image.train(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1

            def before_run(self, run_context):
                self._step += 1
                self._start_time = time.time()
                return tf.train.SessionRunArgs(loss)

            def after_run(self, run_context, run_values):
                duration = time.time() - self._start_time
                loss_value = run_values.results
                if self._step % 10 == 0:
                    num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), self._step,
                                        loss_value, examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.train_dir,
                hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                       tf.train.NanTensorHook(loss),
                       _LoggerHook()],
                config=tf.ConfigProto(
                    log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()
