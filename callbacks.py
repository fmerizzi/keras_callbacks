from tensorflow.keras.callbacks import Callback
import os


# Save the model weights with parametric freq. and folder
class CustomSaveModelCallback(Callback):
    def __init__(self, save_freq = 20, save_path='./'):
        super(CustomSaveModelCallback, self).__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            filename = f'weights_epoch_{epoch + 1}.weights.h5'
            file_path = os.path.join(self.save_path, filename)
            self.model.save_weights(file_path)
            print(f'Saved weights at epoch {epoch + 1} to {file_path}')



# Output callback for when the stdout get lost, making experiments and printing the result on a file
# Kind of a different version of CSVLogger
class OutputCallback(keras.callbacks.Callback):
    def __init__(self, file_name = 'training_output.txt'):
        self.file_name = file_name

    def on_epoch_end(self, epoch, logs=None):
        with open(self.file_name, 'a') as f:
            # obtain the data
            exp, baseline, psnrs, psnrs_baseline, ssims, ssims_baseline = experiment(test_generator,20)
            predmse = np.mean(exp)
            basemse = np.mean(baseline)

            predpsnr = np.mean(psnrs)
            predbasepsnr = np.mean(psnrs_baseline)

            predssim = np.mean(ssims)
            predbasessim = np.mean(ssims_baseline)

            #print the data
            f.write(f'Epoch {epoch + 1}')

            f.write("mse pred: ",predmse)
            f.write("\nmse base: ",basemse)

            f.write("\npsnr pred: ",predpsnr)
            f.write("\npsnr base: ",predbasepsnr)

            f.write("\nssim pred: ",predssim)
            f.write("\nssim base: ",predbasessim)

