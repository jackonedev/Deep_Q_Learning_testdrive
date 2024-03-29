import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

import os

# https://towardsdatascience.com/deep-reinforcement-learning-with-python-part-3-using-tensorboard-to-analyse-trained-models-606c214c14c7

class ModifiedTensorBoard(TensorBoard):
    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = os.path.join(self.log_dir, name)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()
                
                
if __name__ == "__main__":
    


    # # Fit on all samples as one batch, log only on terminal state
    # self.model.fit(x = np.array(X).reshape(-1, *env.ENVIRONMENT_SHAPE),
    #             y = np.array(y),
    #             batch_size = MINIBATCH_SIZE, verbose = 0,
    #             shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

    # # At the start of each episode
    # agent.tensorboard.step = episode


    # # Use this line to update the log file:
    # agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)


    # # visualize the training in tensorboard
    # tensorboard --logdir="logs/"
    
    pass
    