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

    ddp = DDPStrategy(process_group_backend='nccl')

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    postfixname = 'classifier'

    logdir = './logs/2024-11-05_140458_classifier'  # valeur par défaut None

    if logdir is None:  # will be used when there will be an arguments system
        logdir = f'./logs/{now}_{postfixname}'
        lr = 1e-5
        weight_decay = 1e-2
    else:
        best_hp = pd.read_csv(os.path.join(logdir, 'best_hp.csv'))
        lr = best_hp['lr'][0]
        weight_decay = best_hp['weight_decay'][0]

    n_epoch = 1

    batch_size = 30

    patience = 50

    model = MyModel(lr=lr, weight_decay=weight_decay)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/acc",
        dirpath=os.path.join(logdir, 'checkpoints'),
        filename="best_model",
        save_top_k=3,
        mode="min",
    )

    early_stopping_callback = EarlyStopping(
        monitor="val/acc",
        mode="max",
        patience=patience,
        verbose=True
    )

    tensorboard_callback = TensorBoardLogger(name="tensorboard", save_dir=logdir)

    trainer = L.Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=tensorboard_callback,
        max_epochs=n_epoch,
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

    trainer.fit(model, data)


if __name__ == '__main__':
    main()
