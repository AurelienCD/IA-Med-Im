import lightning as L
import optuna
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from data.MRIDataModule_Alexandre import MRIDataModule
from classifieur.mymodel import MyModel
import datetime
import os
import pandas as pd
from lightning.pytorch.strategies import DDPStrategy

def main():

    # TODO :  système arguments
    # TODO : chargement automatique de configuration de modèle via fichier yaml

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    postfixname = 'classifier'

    logdir = './logs/2024-11-07_165557_classifier'  # valeur par défaut None
    #logdir = None

    lr = 1.6319144779067916e-05
    weight_decay = 0.00040378497054698297

    batch_size = 30

    ckptdir = os.path.join(logdir, 'checkpoints')

    model = MyModel(lr=lr, weight_decay=weight_decay)

    tensorboard_callback = TensorBoardLogger(name="tensorboard", save_dir=logdir)

    trainer = L.Trainer(
        max_epochs=100,
        accelerator='gpu',
        devices='1',
        log_every_n_steps=5  # Par defaut log_every_n_steps = 50
    )

    db_path = '../data'
    task = 'classification'
    manifest = f'MRI_dataset_{task}_3classes.csv'

    data = MRIDataModule(
        dataset_path=db_path,
        manifest_filename=manifest,
        batch_size=batch_size,
        task=task,
        crop_size=None,
        train_val_test_shuffle=(True, False, False),
        train_val_test_split=(0.6, 0.2, 0.2),
        weighted_sample=False,
        seed=23,
        verbose=True,
        normalization='max',
        num_workers=None)

    evaluation = {}
    for ckpt in os.listdir(ckptdir):
        evaluation[ckpt] = trainer.test(model, data, ckpt_path=os.path.join(ckptdir, ckpt))[0]
    pd.DataFrame(evaluation).transpose().to_csv(os.path.join(logdir, 'evaluation.csv'))


if __name__ == '__main__':
    main()
