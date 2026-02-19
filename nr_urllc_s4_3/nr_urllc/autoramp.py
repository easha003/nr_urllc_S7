# nr_urllc/autoramp.py
import numpy as np

__all__ = ["AutorampController"]

class AutorampController:
    """
    Fair, modulation-agnostic autoramp controller:
      • Scales min_bits by k = log2(M) so #REs are comparable across M
      • Requires (errors >= target_errs) AND (used_REs >= R_min) AND (bits >= min_bits_eff)
      • Computes fixed sigma^2 from Eb/N0, k, and efficiency eta per chunk
    """

    def __init__(self, cfg: dict, M: int, nfft: int, cp_frac: float, current_snr_db: float = None):
        self.cfg     = cfg
        self.M       = int(M)
        self.k       = int(np.log2(self.M))
        self.nfft    = int(nfft)
        self.cp_frac = float(cp_frac)

        ar_cfg = cfg.get("autoramp", {})
        self.target_errs   = int(ar_cfg.get("target_errs", 200))
        self.base_min_bits = int(ar_cfg.get("base_min_bits", 200_000))
        self.min_used_res  = int(ar_cfg.get("min_used_res", 2_000_000))
        self.max_bits      = int(ar_cfg.get("max_bits", 10_000_000))
        self.noise_blend_w = float(ar_cfg.get("noise_blend_weight", 0.0))  # keep 0.0
    
        base_target = int(ar_cfg.get("target_errs", 200))
        
        # Scale target errors based on SNR (more errors needed in transition region)
        if current_snr_db is not None:
            if 8 <= current_snr_db <= 16:  # Transition region
                self.target_errs = base_target * 3  # 3x more errors
            elif current_snr_db > 16:  # High SNR (low BER)
                self.target_errs = max(base_target // 2, 50)  # Can use fewer
            else:  # Low SNR (high BER)
                self.target_errs = base_target
        else:
            self.target_errs = base_target

            # Scale min_bits so the number of REs is ~constant across M
            # Ref = QPSK -> log2(4) = 2
        self.min_bits_eff = int(self.base_min_bits * (2.0 / max(1, self.k)))

        self.reset_counters()

    # ---------------- counters ----------------
    def reset_counters(self):
        self.total_bits     = 0
        self.total_errs     = 0
        self.used_res_total = 0

    def update_counters(self, bits_tx: np.ndarray, bits_hat: np.ndarray, use_mask) -> None:
        """Update bit errors, bit count, and data-RE coverage."""
        if bits_hat is None or bits_tx is None:
            raise ValueError("AutorampController.update_counters: bits are None")
        err = int(np.count_nonzero(bits_hat != bits_tx))
        self.total_errs     += err
        self.total_bits     += int(bits_tx.size)
        self.used_res_total += int(np.count_nonzero(use_mask))

    def should_stop(self) -> bool:
        have_confidence = (self.total_errs     >= self.target_errs)
        have_coverage   = (self.used_res_total >= self.min_used_res)
        met_min_bits    = (self.total_bits     >= self.min_bits_eff)
        hit_max_bits    = (self.total_bits     >= self.max_bits)
        return (have_confidence and have_coverage and met_min_bits) or hit_max_bits

    # ---------------- SNR → sigma^2 ----------------
    @staticmethod
    def _efficiency_eta(nfft: int, cp_frac: float, n_data_re: int, n_total_re: int) -> float:
        """η ≈ (data-RE fraction) × (nfft / (nfft + cp))."""
        if n_total_re <= 0:
            return 1.0
        data_frac = float(n_data_re) / float(n_total_re)
        cp_eff    = float(nfft) / float(nfft * (1.0 + cp_frac))
        return max(1e-6, data_frac * cp_eff)

    def sigma2_from_ebn0_db(self, ebn0_db: float, eta: float) -> float:
        """
        With Es normalized to 1 per symbol on DATA REs:
          Es/N0 [dB] = Eb/N0 [dB] + 10log10(k) + 10log10(η)
          sigma^2 = 1 / (Es/N0 linear)
        """
        esn0_db = float(ebn0_db) + 10.0*np.log10(max(1, self.k)) + 10.0*np.log10(max(1e-6, eta))
        return 10.0 ** (-esn0_db / 10.0)

    def compute_sigma2_fix(self, ebn0_db: float, use_mask, pilots_mask=None, total_mask=None) -> float:
        """
        Compute per-chunk fixed sigma^2 for MMSE given current RE masks.
        If total_mask is missing, use (data + pilot) counts as total.
        """
        n_data = int(np.count_nonzero(use_mask))
        if total_mask is not None:
            n_total = int(np.count_nonzero(total_mask))
        else:
            n_pil = int(np.count_nonzero(pilots_mask)) if pilots_mask is not None else 0
            n_total = n_data + n_pil

        eta = self._efficiency_eta(self.nfft, self.cp_frac, n_data, n_total)
        return self.sigma2_from_ebn0_db(ebn0_db, eta)
