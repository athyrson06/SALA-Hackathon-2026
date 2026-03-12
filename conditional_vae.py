import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalVAE(nn.Module):
    """
    Causal VAE for precipitation nowcasting with latent confounder Z.
    Matches the style of RecurrentClassifier: configurable via __init__,
    submodules defined inside, forward returns training loss.
    """
    def __init__(self, input_dim, T_in, T_out, latent_dim=16, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.T_in = T_in
        self.T_out = T_out
        self.latent_dim = latent_dim

        # --- Encoder: q(Z | X, Y) ---
        encoder_input_dim = T_in * input_dim + T_out * 1
        self.encoder_net = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        # --- Decoder for X: p(X | Z) ---
        # self.decoder_x_net = nn.Sequential(
        #     nn.Linear(latent_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, T_in * input_dim)
        # )
        self.decoder_x_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, T_in * input_dim)
        )

        # --- Decoder for Y: p(Y | X, Z) ---
        decoder_y_input_dim = T_in * input_dim + latent_dim
        self.decoder_y_net = nn.Sequential(
            nn.Linear(decoder_y_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, T_out)   # single logit per time step
        )
        self.y_pi = nn.Linear(hidden_dim, T_out)      # logit(π)
        self.y_mu = nn.Linear(hidden_dim, T_out)      # raw log(μ)
        self.y_logsigma = nn.Linear(hidden_dim, T_out) # raw log(σ)

    def encode(self, X, Y):
        batch = X.size(0)
        X_flat = X.view(batch, -1)
        Y_flat = Y.view(batch, -1)
        encoder_input = torch.cat([X_flat, Y_flat], dim=1)
        h = self.encoder_net(encoder_input)
        mu_z = self.enc_mu(h)
        logvar_z = self.enc_logvar(h)
        return mu_z, logvar_z

    def decode_x(self, Z):
        X_flat = self.decoder_x_net(Z)
        X = X_flat.view(-1, self.T_in, self.input_dim)
        return X

    def decode_y(self, X, Z):
        batch = X.size(0)
        X_flat = X.view(batch, -1)
        decoder_input = torch.cat([X_flat, Z], dim=1)
        logits = self.decoder_y_net(decoder_input)   # (batch, T_out)
        return logits
    
    # def decode_y(self, X, Z):
    #     batch = X.size(0)
    #     X_flat = X.view(batch, -1)
    #     decoder_input = torch.cat([X_flat, Z], dim=1)
    #     h = self.decoder_y_net(decoder_input)

    #     # --- Stability constraints ---
    #     logit_pi = self.y_pi(h)                       # no constraint needed
    #     log_mu_raw = self.y_mu(h)
    #     log_sigma_raw = self.y_logsigma(h)

    #     # Bound log_mu to [-5, 5] using tanh scaling
    #     log_mu = torch.tanh(log_mu_raw) * 5.0
    #     # Bound log_sigma to [-3, 3] (sigma = exp(log_sigma) in [0.05, 20])
    #     log_sigma = torch.tanh(log_sigma_raw) * 3.0

    #     return logit_pi, log_mu, log_sigma

    def forward(self, X, Y, beta=1.0):
        mu_z, logvar_z = self.encode(X, Y)
        std_z = torch.exp(0.5 * logvar_z)
        eps = torch.randn_like(std_z)
        Z = mu_z + eps * std_z

        X_recon = self.decode_x(Z)
        Y_logits = self.decode_y(X, Z)  # (batch, T_out)

        # --- Losses ---
        recon_loss_x = F.mse_loss(X_recon, X, reduction='mean')

        Y_true_flat = Y.view(-1)  # Flatten to (batch * T_out,)
        Y_logits_flat = Y_logits.view(-1)  # Flatten to same size
        
        recon_loss_y = F.binary_cross_entropy_with_logits(
            Y_logits_flat, Y_true_flat, reduction='mean'
        )

        kl_loss = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp(), dim=1).mean()

        loss = recon_loss_x + recon_loss_y + beta * kl_loss
        return loss, (recon_loss_x, recon_loss_y, kl_loss)

    @torch.no_grad()
    def generate(self, X, num_samples=100, deterministic=False):
        self.eval()
        batch = X.size(0)
        device = X.device

        if deterministic:
            Z_prior = torch.zeros(batch, self.latent_dim, device=device)
            logits = self.decode_y(X, Z_prior)          # (batch, T_out)
            probs = torch.sigmoid(logits)
            return probs.unsqueeze(0).unsqueeze(-1)     # shape (1, batch, T_out, 1)
        else:
            Z_prior = torch.randn(num_samples, batch, self.latent_dim, device=device)
            X_exp = X.unsqueeze(0).expand(num_samples, -1, -1, -1)
            samples = []
            for i in range(num_samples):
                logits = self.decode_y(X_exp[i], Z_prior[i])
                probs = torch.sigmoid(logits)
                # Bernoulli sample
                y_sample = torch.bernoulli(probs).unsqueeze(-1)
                samples.append(y_sample)
            return torch.stack(samples)   # (num_samples, batch, T_out, 1)

    # def forward(self, X, Y, beta=1.0):
    #     mu_z, logvar_z = self.encode(X, Y)
    #     std_z = torch.exp(0.5 * logvar_z)
    #     eps = torch.randn_like(std_z)
    #     Z = mu_z + eps * std_z

    #     X_recon = self.decode_x(Z)
    #     logit_pi, log_mu, log_sigma = self.decode_y(X, Z)

    #     # ---- Losses ----
    #     recon_loss_x = F.mse_loss(X_recon, X, reduction='sum') / X.size(0)

    #     Y_true = Y.squeeze(-1)                     # (batch, T_out)
    #     pi = torch.sigmoid(logit_pi)
    #     mu = torch.exp(log_mu)                      # mean of log(Y)
    #     sigma = torch.exp(log_sigma) + 1e-6         # ensure positive

    #     normal = torch.distributions.LogNormal(mu, sigma)
    #     log_prob_pos = normal.log_prob(Y_true + 1e-6)
    #     log_prob = torch.where(Y_true == 0,
    #                            torch.log(pi + 1e-6),
    #                            torch.log(1 - pi + 1e-6) + log_prob_pos)
    #     recon_loss_y = -log_prob.sum(dim=1).mean()

    #     kl_loss = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp(), dim=1).mean()

    #     loss = recon_loss_x + recon_loss_y + beta * kl_loss
    #     return loss, (recon_loss_x, recon_loss_y, kl_loss)
    
    # @torch.no_grad()
    # def generate(self, X, num_samples=100, deterministic=False):
    #     self.eval()
    #     batch = X.size(0)
    #     device = X.device
        
    #     if deterministic:
    #         # 1. Use the mean of the latent prior (Z = 0) for a stable, deterministic output
    #         Z_prior = torch.zeros(batch, self.latent_dim, device=device)
            
    #         # 2. Decode using the deterministic Z
    #         # Note: We don't need X_exp here since we only have 1 deterministic pass
    #         logit_pi, log_mu, log_sigma = self.decode_y(X, Z_prior)
            
    #         pi = torch.sigmoid(logit_pi)
    #         mu = torch.exp(log_mu)
    #         sigma = torch.exp(log_sigma) + 1e-6
            
    #         # 3. Calculate the theoretical expected value
    #         # E[Y] = (1 - pi) * exp(mu + (sigma^2) / 2)
    #         expected_pos = torch.exp(mu + (sigma ** 2) / 2.0)
    #         expected_y = (1.0 - pi) * expected_pos
            
    #         # Return with dummy sample dimension (1) and final dimension to match your setup
    #         # Shape: (1, batch, 1) or similar, depending on expected_y's base shape
    #         return expected_y.unsqueeze(0).unsqueeze(-1)
            
    #     else:
    #         # --- YOUR ORIGINAL STOCHASTIC CODE ---
    #         Z_prior = torch.randn(num_samples, batch, self.latent_dim, device=device)
    #         X_exp = X.unsqueeze(0).expand(num_samples, -1, -1, -1)

    #         Y_samples = []
    #         for i in range(num_samples):
    #             logit_pi, log_mu, log_sigma = self.decode_y(X_exp[i], Z_prior[i])
    #             pi = torch.sigmoid(logit_pi)
    #             mu = torch.exp(log_mu)
    #             sigma = torch.exp(log_sigma) + 1e-6
    #             zero_mask = torch.bernoulli(pi) == 1
    #             pos_samples = torch.exp(torch.normal(mu, sigma))
    #             y_sample = torch.where(zero_mask, torch.zeros_like(pos_samples), pos_samples)
    #             Y_samples.append(y_sample.unsqueeze(-1))
                
    #         return torch.stack(Y_samples)