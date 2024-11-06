import lightning as L
import optuna
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from data.MRIDataModule_Alexandre import MRIDataModule
from classifieur.mymodel import MyModel
import datetime
import os
import pandas as pd

def main():

    # TODO : système arguments
    # TODO : chargement automatique de configuration de modèle via fichier yaml

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    postfixname = 'classifier'

    logdir = f'./logs/{now}_{postfixname}'

    n_epoch = 300

    n_trials = 50

    batch_size = 30

    min_lr = 1e-5
    max_lr = 1e-2

    min_weight_decay = 1e-6
    max_weight_decay = 1e-2

    patience = 50

    def objective(trial):
        lr = trial.suggest_float('lr', min_lr, max_lr, log=True)
        weight_decay = trial.suggest_float('weight_decay', min_weight_decay, max_weight_decay, log=True)

        print(f'\ntrial : {trial}')
        print(f'lr : {lr}')
        print(f'weight_decay : {weight_decay}\n')

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

        #tensorboard_callback = TensorBoardLogger(name="tensorboard", save_dir=logdir)

        trainer = L.Trainer(
            callbacks=[checkpoint_callback, early_stopping_callback],
            #logger=tensorboard_callback,
            max_epochs=n_epoch,
            accelerator='gpu',
            devices='1',
            log_every_n_steps=5   #Par defaut log_every_n_steps = 50
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
        val_acc = trainer.callback_metrics['val/acc'].item()
        return val_acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("Best hyperparameters:", study.best_params)

    best_hp = pd.DataFrame([{'lr': study.best_params['lr'], 'weight_decay': study.best_params['weight_decay']}])
    best_hp.to_csv(os.path.join(logdir, 'best_hp.csv'))

if __name__ == '__main__':
    main()
