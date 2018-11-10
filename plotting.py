import matplotlib.pyplot as plt

def plotHistory(history):
    """Plot training/validation accuracy and loss
    """
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.legend(['train','val'],loc='lower right')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.legend(['train','val'],loc='upper right')
    plt.show()
