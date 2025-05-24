import os, json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers  # Layers
from tensorflow.keras.optimizers import Adam # Optimizer 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from utils.models import *
from utils.utils_augm import *

#---------------------------------------------------------------------------

def get_actual_predicted_labels(dataset, model, save_errors=False, path_output=None):
    """
      Create a list of actual ground truth values and the predictions from the model.

      Args:
        dataset: An iterable data structure, such as a TensorFlow Dataset, with features and labels.

      Return:
        Ground truth and predicted values for a particular dataset.
    """
    actual = [labels for _, labels in dataset.unbatch()]
    videos = [vid for vid, _ in dataset.unbatch()]
    actual = tf.argmax(actual, axis=1)
    res = model.predict(dataset)

    actual = tf.stack(actual, axis=0)
    proba = tf.concat(res, axis=0)
    predicted = tf.argmax(proba, axis=1)
    
    if save_errors==True and path_output!=None:
        os.makedirs(path_output, exist_ok=True)
        for i, win in enumerate(videos):
            if actual[i]!=predicted[i]:
                outfile = path_output + "C{}_P{}_{}.avi".format(actual[i], predicted[i], i)
                out = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'MJPG'), 1, (win.shape[1], win.shape[2]), True)
                for k in range(0,win.shape[0]):
                    #plt.imshow(win[k,...])
                    #plt.show()
                    data = np.array(win[k,...]*255., dtype=np.uint8)[..., [2, 1, 0]]
                    out.write(data)
                out.release()

    return actual, predicted, proba

#---------------------------------------------------------------------------
                     
def plot_history(history, results_path):
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.semilogy(history.history['loss'], label='Train')
    plt.semilogy(history.history['val_loss'], label='Validation')
    plt.grid(True)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.subplot(1,2,2)
    plt.semilogy(history.history['accuracy'], label='Train')
    plt.semilogy(history.history['val_accuracy'], label='Validation')
    plt.grid(True)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig(os.path.join(results_path, 'history.png'), bbox_inches = "tight", dpi=200)

#---------------------------------------------------------------------------

def train_model(model, train_generator, val_generator, BATCH_SIZE, WEIGHTS_DIR, epochs):
    model_checkpoint = ModelCheckpoint(WEIGHTS_DIR+os.path.sep+'classifier.hdf5', monitor='val_accuracy', verbose=0, save_best_only=True)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate = 0.001), metrics=["accuracy"]) # bisognerebbe usare binary
    weight_for_0 = 0.4
    weight_for_1 = 0.6
    history=model.fit(
                      train_generator,
                      epochs = epochs,
                      batch_size = BATCH_SIZE,
                      verbose = 1,
                      validation_data = val_generator,
                      callbacks = model_checkpoint,
                      class_weight= {0: weight_for_0, 1: weight_for_1}
                     )
    
    return history

#---------------------------------------------------------------------------

def main(args):
    
    tf.keras.backend.clear_session()
    DATA_PATH = {'train':pathlib.Path(args.training_path),
                 'validation':pathlib.Path(args.validation_path)}
    WEIGHTS_DIR = args.weights_path
    os.makedirs(WEIGHTS_DIR+'/errors/', exist_ok=True)
    os.makedirs(WEIGHTS_DIR+'/errors/train/', exist_ok=True)
    os.makedirs(WEIGHTS_DIR+'/errors/validation/', exist_ok=True)
    
    BATCH_SIZE = 4
    N_CLASSES = 2
    EPOCHS = 100
    WINDOW_SIZE = (20,50,50,3) 

    output_signature = (tf.TensorSpec(shape = (None, None, None, WINDOW_SIZE[-1]), dtype = tf.float32),
                        tf.TensorSpec(shape = (N_CLASSES,), dtype = tf.float32))
    train_ds = tf.data.Dataset.from_generator(FrameGenerator(DATA_PATH['train'], WINDOW_SIZE[0], WINDOW_SIZE[1], 
                                                             training = True, augmentation=True, undersample_factor=0),
                                              output_signature = output_signature)
    train_ds = train_ds.batch(BATCH_SIZE).shuffle(10).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_generator(FrameGenerator(DATA_PATH['validation'], WINDOW_SIZE[0], WINDOW_SIZE[1]),
                                            output_signature = output_signature)

    val_ds = val_ds.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
    
    model = conv_3D(WINDOW_SIZE, N_CLASSES)
    model.summary()
    
    #Fit the model
    history = train_model(model, train_ds, val_ds, BATCH_SIZE, WEIGHTS_DIR, EPOCHS)         
    plot_history(history, WEIGHTS_DIR)
    model = tf.keras.models.load_model(WEIGHTS_DIR+os.path.sep+'classifier.hdf5')
    
    train_ds = tf.data.Dataset.from_generator(FrameGenerator(DATA_PATH['train'], WINDOW_SIZE[0], WINDOW_SIZE[1], training = False),
                                              output_signature = output_signature)
    train_ds = train_ds.batch(BATCH_SIZE)
    print("Training: ", model.evaluate(train_ds))
    print("Validation: ", model.evaluate(val_ds))
    actual, predicted, proba = get_actual_predicted_labels(train_ds, model, save_errors=True, path_output=WEIGHTS_DIR+'/errors/train/')
    gtlabels = np.unique(actual).astype(str)
    plot_confusion_matrix(actual, predicted, gtlabels, 'training', WEIGHTS_DIR)
    plot_auc(actual, proba, WEIGHTS_DIR, 'training')

    actual, predicted, proba = get_actual_predicted_labels(val_ds, model, save_errors=True, path_output=WEIGHTS_DIR+'/errors/validation/')
    plot_confusion_matrix(actual, predicted, gtlabels, 'validation', WEIGHTS_DIR)
    plot_auc(actual, proba, WEIGHTS_DIR, 'validation')
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Training CVBCC")
    
    parser.add_argument('--training_path', required=True, help='Path to the training video (.avi file)')
    parser.add_argument('--validation_path', required=True, help='Path to the validation video (.avi file)')
    parser.add_argument('--weights_path', required=True, help='Directory to save the model weights')

    args = parser.parse_args()
    main(args)
    
