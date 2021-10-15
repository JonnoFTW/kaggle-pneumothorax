from unetplusplus import Xnet

from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard, LambdaCallback
import keras.backend as K
from keras.layers import Input

from datetime import datetime

from lrf_on_plateau import LRFOnPlateau
from metrics import dice_coef, mean_iou, iou, dice_coef_loss, jaccard_coef_logloss, tversky_loss
import tensorflow as tf
import pickle


def get_loss():
    # func = tversky_loss()
    # func.__name__ = "tversky_loss"
    func = dice_coef_loss
    func.__name__ = "dice_loss"
    return func


class UnetPlusPlusModel:
    def __init__(self, input_size=(512, 512), batch_size=8):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        K.set_session(tf.Session(config=config))
        self.batch_size = batch_size
        self.input_size = input_size
        if len(self.input_size) == 2:
            self.input_size += (3,)
        self.built = False

    def build(self):
        model = Xnet(
            backbone_name='resnet34',
            decoder_filters=(512, 256, 128, 64, 32),
            decoder_use_batchnorm=True,
            input_shape=self.input_size
        )
        metrics = [
            mean_iou,
            iou,
            dice_coef
        ]
        self.loss_func = get_loss()
        # model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=metrics)
        model.compile(optimizer=Adam(),
                      loss=self.loss_func,
                      metrics=metrics)
        print(model.summary())
        self.model = model
        self.tensorboard = TensorBoard(f'logs/unetpp_diceloss_{datetime.now().timestamp()}',
                                       update_freq=128,
                                       write_graph=True)
        self.model_checkpoint = ModelCheckpoint(f'models/unetpp_diceloss_{datetime.now().timestamp()}.hdf5',
                                                monitor='val_loss',
                                                mode='min', verbose=1, save_best_only=True)
        self.built = True

    def fit(self,
            train_gen,
            batch_size,
            num_samples,
            validation_samples,
            validation_gen,
            epochs=1):
        # potentially train two models
        # one for left lung, one for right lung
        if not self.built:
            self.build()
        print(f"Training for {epochs} with steps_per_epoch={num_samples // batch_size}")

        lrfopcb = LRFOnPlateau(train_gen, num_samples, batch_size, min_lr=0.00005, max_lr=0.1,patience=10, epochs=2, cooldown=0,verbose=1)
        return self.model.fit_generator(train_gen,
                                        callbacks=[
                                            self.model_checkpoint,
                                            self.tensorboard,
                                            lrfopcb
                                        ],
                                        epochs=epochs,
                                        verbose=1,
                                        validation_data=validation_gen,
                                        validation_steps=validation_samples // batch_size,
                                        steps_per_epoch=num_samples // batch_size)

    def predict(self, images):
        return self.model.predict(images)

    def save(self):
        fname = f'models/unetpp_dice_{int(datetime.now().timestamp())}.h5'
        print(f"Saving model as {fname}")
        with open(fname + '.misc.pkl', 'wb') as fh:
            pickle.dump({
                'model_cb': {'filepath': self.model_checkpoint.filepath,
                             'monitor': self.model_checkpoint.monitor,
                             'mode': 'min',
                             'verbose': 1,
                             'save_best_only': True},
                'tb_cb': {'log_dir': self.tensorboard.log_dir,
                          'update_freq': self.tensorboard.update_freq,
                          'write_graph': self.tensorboard.write_graph}
            }, fh)
        return self.model.save(fname)

    def load(self, fname):
        with open(fname + '.misc.pkl', 'rb') as fh:
            obj = pickle.load(fh)
            self.model_checkpoint = ModelCheckpoint(**obj['model_cb'])
            self.tensorboard = TensorBoard(**obj['tb_cb'])
        self.model = load_model(fname, custom_objects={
            'tvsersky_loss': get_loss(),
            'mean_iou': mean_iou,
            'iou': iou,
            'dice_coef': dice_coef,
        })

    def set_lr(self, new_lr):
        print("Setting learning rate to", new_lr)
        return K.set_value(self.model.optimizer.lr, new_lr)

    def get_lr(self):
        return K.get_value(self.model.optimizer.lr)
