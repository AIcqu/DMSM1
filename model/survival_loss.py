import tensorflow as tf
from tensorflow.keras import backend as k

def logloss(lambda3):
    def loss(y_true, y_pred):
        mask_dead = y_true[:, 1]
        mask_alive = y_true[:, 0]
        mask_censored = 1 - (mask_alive + mask_dead)
        logloss = -1 * k.mean(mask_dead * k.log(y_pred[:, 1]) + mask_alive * k.log(y_pred[:, 0]))
        - lambda3 * k.mean(y_pred[:, 1] * mask_censored * k.log(y_pred[:, 1]))
        return logloss

    return loss

def rankingloss(y_true, y_pred, time_length,name=None):
    ranking_loss = 0
    for i in range(time_length):
        for j in range(i + 1, time_length, 1):
            tmp1  = y_pred[:, j] - y_pred[:, i]
            tmp2 = y_true[:, j] - y_true[:, i]
            tmp=tf.cast(tmp2,tf.float32)-tf.cast(tmp1,tf.float32)
            tmp3=tmp>0
            tmp3=tf.cast(tmp3, tf.float32)
            ranking_loss=ranking_loss+k.mean(k.square((tmp3*tmp)))

    return ranking_loss