import numpy as np
from tensorflow import keras
from keras import backend as K
from keras.callbacks import Callback

def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
   """
   参数：
   		global_step: 上面定义的Tcur，记录当前执行的步数。
   		learning_rate_base：预先设置的学习率，当warm_up阶段学习率增加到learning_rate_base，就开始学习率下降。
   		total_steps: 是总的训练的步数，等于epoch*sample_count/batch_size,(sample_count是样本总数，epoch是总的循环次数)
   		warmup_learning_rate: 这是warm up阶段线性增长的初始值
   		warmup_steps: warm_up总的需要持续的步数
   		hold_base_rate_steps: 这是可选的参数，即当warm up阶段结束后保持学习率不变，知道hold_base_rate_steps结束后才开始学习率下降
   """
   if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    #这里实现了余弦退火的原理，设置学习率的最小值为0，所以简化了表达式
   learning_rate = 0.5 * learning_rate_base * (1 + np.cos(np.pi *
        (global_step - warmup_steps - hold_base_rate_steps) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    #如果hold_base_rate_steps大于0，表明在warm up结束后学习率在一定步数内保持不变
   if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
   if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        #线性增长的实现
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        #只有当global_step 仍然处于warm up阶段才会使用线性增长的学习率warmup_rate，否则使用余弦退火的学习率learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
   return np.where(global_step > total_steps, 0.0, learning_rate)


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    """
    继承Callback，实现对学习率的调度
    """
    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        #learning_rates用于记录每次更新后的学习率，方便图形化观察
        self.learning_rates = []
	#更新global_step，并记录当前学习率
    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)
	#更新学习率
    def on_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))


class SnapshotEnsemble(Callback):

    def __init__(self, n_epochs, n_cycles, lrate_max, verbose=0):
        self.epochs = n_epochs
        self.cycles = n_cycles
        self.lr_max = lrate_max
        self.lrates = list()

    def cosine_annealing(self, epoch, n_epochs, n_cycles, lrate_max):
        epochs_per_cycle = n_epochs // n_cycles
        if epoch<10:
            self.lr_max = lrate_max
            return lrate_max
        elif epoch==10:
            lr = lrate_max*10
            self.lr_max = lr
            return lr
        elif epoch>10 and epoch % epochs_per_cycle==0:
            lr = lrate_max*3
            self.lr_max = lr
            return self.lr_max
        elif epoch>10:
            cos_inner = (np.pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
            lr = lrate_max * 0.8
            self.lr_max = lr
            return self.lr_max

    def on_epoch_begin(self, epoch, logs={}):
        lr = self.cosine_annealing(epoch, self.epochs, self.cycles, self.lr_max)
        print(f'epoch {epoch+1}, lr {lr}')
        K.set_value(self.model.optimizer.lr, lr)
        self.lrates.append(lr)

    def on_epoch_end(self, epoch,logs={}):
        # epochs_per_cycle = self.epochs // self.cycles
        # if epoch != 0 and (epoch + 1) % epochs_per_cycle == 0:
        #     filename = f"snapshot_model_{int((epoch+1) / epochs_per_cycle)}.h5"
        #     self.model.save(filename)
        #     print(f'>saved snapshot {filename}, epoch {epoch}')
        pass

class LearningRateScheduler(Callback):
    def __init__(self, n_epochs, verbose=0):
        self.epochs = n_epochs
        self.lrates = list()

    def lr_scheduler(self, epoch, n_epochs):
        initial_lrate = 0.1
        lrate = initial_lrate * np.exp(-0.02*epoch)
        return lrate

    def on_epoch_begin(self, epoch, logs={}):
        lr = self.lr_scheduler(epoch, self.epochs)
        print(f'epoch {epoch+1}, lr {lr}')
        K.set_value(self.model.optimizer.lr, lr)
        self.lrates.append(lr)
