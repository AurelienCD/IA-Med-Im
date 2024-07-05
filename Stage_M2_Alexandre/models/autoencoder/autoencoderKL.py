import torch
import lightning as L
from models.autoencoder.encoder import Encoder
from models.autoencoder.decoder import Decoder
from models.autoencoder.building_blocks import DiagonalGaussianDistribution
from utils.autoloading import instantiate_from_config


class AutoEncoderKL(L.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 monitor=None,
                 ):
        super().__init__()
        self.automatic_optimization = False
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        """
        posterior: output of encoder (latent variables)
        dec: output of decoder (reconstruct image)
        """
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch):
        """
        define reformating input if needed
        """
        return batch

    def training_step(self, batch, batch_idx):
        inputs = self.get_input(batch)
        reconstructions, posterior = self(inputs)
        opt_ae, opt_disc = self.optimizers()

        """
        -----   Optimize Autoencoder   -----
        """
        # train encoder+decoder+logvar
        self.toggle_optimizer(opt_ae)

        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx=0,
                                        global_step=self.global_step, last_layer=self.get_last_layer(),
                                        split="train")

        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=True)

        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        opt_ae.step()
        self.untoggle_optimizer(opt_ae)

        """
        -----   Optimize Discriminator -----
        """
        # train the discriminator
        if (batch_idx + 1) % 2 == 0:  # optimize discriminant every two steps
            self.toggle_optimizer(opt_disc)

            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx=1,
                                                global_step=self.global_step, last_layer=self.get_last_layer(),
                                                split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=True)

            opt_disc.zero_grad()
            self.manual_backward(discloss)
            opt_disc.step()
            self.untoggle_optimizer(opt_disc)

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        # self.log("val/rec_loss", log_dict_ae["val/rec_loss"], sync_dist=True)
        self.log_dict(log_dict_ae, sync_dist=True)
        self.log_dict(log_dict_disc, sync_dist=True)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return opt_ae, opt_disc

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch)
        if not only_inputs:
            xrec, posterior = self(x)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log
