
# load used library
import os
import matplotlib.pyplot as plt


def plot_training(history, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    acc = history.history.get('accuracy') or history.history.get('acc')
    val_acc = history.history.get('val_accuracy') or history.history.get('val_acc')
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))
    
    # Accuracy
    plt.figure()
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'c', label='Validation acc')
    plt.legend()
    plt.title('Training and validation accuracy')
    plt.savefig(os.path.join(save_dir, 'accuracy.png'))
    plt.close()

    # Loss
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'c', label='Validation loss')
    plt.legend()
    plt.title('Training and validation loss')
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    plt.close()
