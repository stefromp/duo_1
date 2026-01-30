"""
Frequency-Informed Masking for Masked Diffusion Language Models

Based on: "Masked Diffusion Language Models with Frequency-Informed Training"
(Kosmopoulou et al., 2025) - arXiv:2509.05056

This module implements frequency-informed masking strategies that prioritize
learning from rare tokens while maintaining theoretical validity of the
diffusion objective.

Key features:
1. Frequency-based token weighting: Rare tokens get higher masking probability
2. Curriculum learning: Progressively increase focus on rare tokens during training
3. Multiple noise scheduling strategies: Cosine, Gaussian, Bimodal Gaussian
4. ELBO derivative softening for improved performance with certain schedules
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Optional, Dict, Union, Any, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    import datasets


# =============================================================================
# Token Frequency Computation
# =============================================================================

def compute_token_frequencies(
    dataset_or_texts: Union[list, Any],
    tokenizer,
    text_column: str = "text",
    max_samples: Optional[int] = None,
) -> torch.Tensor:
    """Compute token frequencies from a dataset.
    
    Args:
        dataset_or_texts: Either a list of texts or a HuggingFace dataset
        tokenizer: The tokenizer to use
        text_column: Column name for text data in HuggingFace datasets
        max_samples: Maximum number of samples to process (for efficiency)
    
    Returns:
        freq_tensor: Tensor of shape (vocab_size,) with token frequencies
    """
    counter = Counter()
    vocab_size = tokenizer.vocab_size
    
    if hasattr(dataset_or_texts, "__iter__") and not isinstance(dataset_or_texts, str):
        # Process samples
        samples = dataset_or_texts
        if max_samples is not None:
            samples = list(samples)[:max_samples]
        
        for item in samples:
            if isinstance(item, dict):
                text = item.get(text_column, "")
            else:
                text = str(item)
            
            tokens = tokenizer.encode(text, add_special_tokens=False)
            counter.update(tokens)
    
    # Convert to frequency tensor
    total = sum(counter.values()) + 1e-10  # Avoid division by zero
    freq_tensor = torch.zeros(vocab_size)
    for token_id, count in counter.items():
        if token_id < vocab_size:
            freq_tensor[token_id] = count / total
    
    # Add small epsilon for tokens not seen
    freq_tensor = freq_tensor + 1e-10
    freq_tensor = freq_tensor / freq_tensor.sum()  # Renormalize
    
    return freq_tensor


def compute_frequency_weights_from_tokenizer(tokenizer) -> torch.Tensor:
    """Compute approximate frequency weights based on Zipf's law.
    
    This is a simpler alternative when you don't have access to the 
    training dataset. Assumes token IDs roughly correspond to frequency rank.
    
    Args:
        tokenizer: The tokenizer to use
    
    Returns:
        weights: Tensor of shape (vocab_size,) with rarity weights (higher = rarer)
    """
    vocab_size = tokenizer.vocab_size
    
    # Use Zipf's law approximation: frequency ∝ 1/rank
    # Rank is roughly approximated by token ID for most tokenizers
    ranks = torch.arange(1, vocab_size + 1, dtype=torch.float32)
    frequencies = 1.0 / ranks
    
    # Convert to rarity weights (inverse of frequency)
    # Higher weight = rarer token
    rarity_weights = 1.0 / (frequencies + 1e-10)
    
    # Normalize to [0, 1]
    rarity_weights = (rarity_weights - rarity_weights.min()) / (
        rarity_weights.max() - rarity_weights.min() + 1e-10
    )
    
    return rarity_weights


# =============================================================================
# Frequency-Informed Masking
# =============================================================================

class FrequencyInformedMasking(nn.Module):
    """Frequency-informed masking that biases masking toward rare tokens.
    
    This implements the method from Kosmopoulou et al. (2025):
    1. Rank tokens by global frequency (rarer = higher rank)
    2. Normalize ranks to weights in (0, 1)
    3. Soften weights by raising to power p < 1
    4. Scale weights so their mean equals target masking rate (1 - α_t)
    
    The curriculum learning option progressively increases p from 0 to p_target,
    making the distribution sharper over time.
    """
    
    def __init__(
        self,
        vocab_size: int,
        frequency_weights: Optional[torch.Tensor] = None,
        softening_power: float = 0.02,
        use_curriculum: bool = True,
        curriculum_start_epoch: int = 0,
        curriculum_end_epoch: int = 10,
        mask_index: Optional[int] = None,
        special_tokens: Optional[list] = None,
    ):
        """Initialize frequency-informed masking.
        
        Args:
            vocab_size: Size of vocabulary
            frequency_weights: Precomputed rarity weights (shape: vocab_size)
                              Higher values = rarer tokens = more likely to mask
            softening_power: Power p to raise weights to (default: 0.02)
            use_curriculum: Whether to use curriculum learning
            curriculum_start_epoch: Epoch to start curriculum
            curriculum_end_epoch: Epoch to reach full softening_power
            mask_index: Index of mask token (won't be used as target)
            special_tokens: List of special token IDs to never prioritize
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.softening_power_target = softening_power
        self.use_curriculum = use_curriculum
        self.curriculum_start_epoch = curriculum_start_epoch
        self.curriculum_end_epoch = curriculum_end_epoch
        self.mask_index = mask_index
        self.special_tokens = special_tokens or []
        
        # Register frequency weights as buffer (not trained)
        if frequency_weights is None:
            # Default: uniform weights (no frequency bias)
            frequency_weights = torch.ones(vocab_size)
        
        # Normalize to [0, 1]
        frequency_weights = (frequency_weights - frequency_weights.min()) / (
            frequency_weights.max() - frequency_weights.min() + 1e-10
        )
        
        # Set special tokens to low weight (don't prioritize masking them)
        for token_id in self.special_tokens:
            if token_id is not None and token_id < vocab_size:
                frequency_weights[token_id] = 0.0
        if mask_index is not None and mask_index < vocab_size:
            frequency_weights[mask_index] = 0.0
        
        self.register_buffer("base_weights", frequency_weights)
        self._current_epoch = 0
    
    def set_epoch(self, epoch: int):
        """Update current epoch for curriculum learning."""
        self._current_epoch = epoch
    
    def _get_current_power(self) -> float:
        """Compute current softening power based on curriculum."""
        if not self.use_curriculum:
            return self.softening_power_target
        
        if self._current_epoch < self.curriculum_start_epoch:
            return 0.0  # Uniform masking
        elif self._current_epoch >= self.curriculum_end_epoch:
            return self.softening_power_target
        else:
            # Linear interpolation
            progress = (self._current_epoch - self.curriculum_start_epoch) / (
                self.curriculum_end_epoch - self.curriculum_start_epoch
            )
            return progress * self.softening_power_target
    
    def compute_masking_probs(
        self,
        x: torch.Tensor,
        target_mask_rate: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token masking probabilities.
        
        Args:
            x: Input token IDs, shape (batch_size, seq_length)
            target_mask_rate: Target average masking rate (1 - α_t),
                             shape (batch_size, 1) or scalar
        
        Returns:
            mask_probs: Per-token masking probability, shape (batch_size, seq_length)
        """
        batch_size, seq_length = x.shape
        device = x.device
        
        # Get weights for each token in the sequence
        # base_weights: (vocab_size,) -> gather to (batch_size, seq_length)
        weights = self.base_weights.to(device)[x]  # (B, L)
        
        # Get current softening power
        p = self._get_current_power()
        
        if p == 0.0:
            # Uniform masking (no frequency bias)
            return target_mask_rate.expand(batch_size, seq_length)
        
        # Soften weights by raising to power p
        # This prevents over-emphasis on extremely rare tokens
        softened_weights = weights ** p  # (B, L)
        
        # Ensure target_mask_rate has right shape
        if target_mask_rate.ndim == 0:
            target_mask_rate = target_mask_rate.unsqueeze(0).unsqueeze(0)
        elif target_mask_rate.ndim == 1:
            target_mask_rate = target_mask_rate.unsqueeze(-1)
        target_mask_rate = target_mask_rate.expand(batch_size, 1)  # (B, 1)
        
        # Conditional scaling to ensure mean equals target rate
        # From the paper (Equation 2):
        # w_new = w^p * (1 - α_t) / μ    if μ > 1 - α_t
        # w_new = -(1 - w^p) * α_t / (1 - μ) + 1   otherwise
        
        mu = softened_weights.mean(dim=1, keepdim=True)  # (B, 1)
        
        # Case 1: μ > target_rate -> scale down
        # Case 2: μ <= target_rate -> scale up
        scale_down = mu > target_mask_rate
        
        mask_probs = torch.where(
            scale_down,
            softened_weights * target_mask_rate / (mu + 1e-10),
            1 - (1 - softened_weights) * (1 - target_mask_rate) / (1 - mu + 1e-10)
        )
        
        # Clamp to valid probability range
        mask_probs = mask_probs.clamp(0.0, 1.0)
        
        return mask_probs
    
    def sample_mask(
        self,
        x: torch.Tensor,
        target_mask_rate: torch.Tensor,
    ) -> torch.Tensor:
        """Sample binary mask using frequency-informed probabilities.
        
        Args:
            x: Input token IDs, shape (batch_size, seq_length)
            target_mask_rate: Target masking rate (1 - α_t)
        
        Returns:
            mask: Binary mask, shape (batch_size, seq_length)
                  1 = token should be masked, 0 = keep original
        """
        mask_probs = self.compute_masking_probs(x, target_mask_rate)
        mask = torch.rand_like(mask_probs) < mask_probs
        return mask


# =============================================================================
# Noise Schedules
# =============================================================================

class CosineSchedule(nn.Module):
    """Cosine noise schedule: α_t = cos(π/2 * (1 - t))
    
    This schedule concentrates masking rates at lower values,
    with an expected mean of ~0.36 instead of 0.5 for linear.
    """
    
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps
    
    def forward(self, t: torch.Tensor):
        """Compute noise schedule values.
        
        Args:
            t: Time values in [0, 1], shape (batch_size,) or (batch_size, 1)
        
        Returns:
            dalpha_t: Time derivative of α_t
            alpha_t: Signal level α_t
        """
        t = t.clamp(self.eps, 1.0 - self.eps)
        
        # α_t = cos(π/2 * (1 - t))
        alpha_t = torch.cos(math.pi / 2 * (1 - t))
        
        # d(α_t)/dt = π/2 * sin(π/2 * (1 - t))
        dalpha_t = math.pi / 2 * torch.sin(math.pi / 2 * (1 - t))
        
        return dalpha_t, alpha_t


class GaussianSchedule(nn.Module):
    """Gaussian (unimodal) noise schedule.
    
    The masking rate (1 - α_t) is sampled from a Gaussian distribution
    when t is sampled uniformly.
    """
    
    def __init__(
        self,
        mean: float = 0.3,
        std: float = 0.1,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.mean = mean
        self.std = std
        self.eps = eps
    
    def forward(self, t: torch.Tensor):
        """Compute noise schedule values.
        
        Args:
            t: Time values in [0, 1]
        
        Returns:
            dalpha_t: Time derivative of α_t  
            alpha_t: Signal level α_t
        """
        # Map uniform t to Gaussian masking rate
        # Using inverse CDF of Gaussian
        from torch.distributions import Normal
        normal = Normal(self.mean, self.std)
        
        # Clamp t to avoid infinities at boundaries
        t = t.clamp(self.eps, 1.0 - self.eps)
        
        # Masking rate = Φ^{-1}(t) mapped through Gaussian
        mask_rate = normal.icdf(t)
        mask_rate = mask_rate.clamp(self.eps, 1.0 - self.eps)
        
        alpha_t = 1 - mask_rate
        
        # Approximate derivative numerically
        delta = 1e-4
        mask_rate_plus = normal.icdf((t + delta).clamp(self.eps, 1.0 - self.eps))
        dalpha_t = -(mask_rate_plus - mask_rate) / delta
        
        return dalpha_t, alpha_t


class BimodalGaussianSchedule(nn.Module):
    """Bimodal Gaussian noise schedule with time-varying right mode.
    
    The masking rate is sampled from a mixture of two Gaussians:
    - Left mode: Concentrated at low masking rates (fine-grained learning)
    - Right mode: Moves to higher masking rates over training (coarse learning)
    
    This creates a natural curriculum from easy to hard denoising.
    """
    
    def __init__(
        self,
        left_weight: float = 0.6,
        left_mean: float = 0.12,
        left_std: float = 0.02,
        right_mean_start: float = 0.4,
        right_mean_end: float = 0.85,
        right_std: float = 0.08,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.left_weight = left_weight
        self.left_mean = left_mean
        self.left_std = left_std
        self.right_mean_start = right_mean_start
        self.right_mean_end = right_mean_end
        self.right_std = right_std
        self.eps = eps
        self._training_progress = 0.0  # 0 to 1
    
    def set_training_progress(self, progress: float):
        """Update training progress (0 = start, 1 = end)."""
        self._training_progress = min(max(progress, 0.0), 1.0)
    
    def _get_right_mean(self) -> float:
        """Get current right mode mean based on training progress."""
        # Exponential transition: μ_2(τ) = 0.4 + (0.85 - 0.4) * (1 - e^{-τ})
        tau = self._training_progress * 3  # Scale for faster transition
        return self.right_mean_start + (
            self.right_mean_end - self.right_mean_start
        ) * (1 - math.exp(-tau))
    
    def forward(self, t: torch.Tensor):
        """Compute noise schedule values.
        
        Args:
            t: Time values in [0, 1]
        
        Returns:
            dalpha_t: Time derivative of α_t
            alpha_t: Signal level α_t
        """
        from torch.distributions import Normal, MixtureSameFamily, Categorical
        
        t = t.clamp(self.eps, 1.0 - self.eps)
        
        # Create mixture distribution
        right_mean = self._get_right_mean()
        
        # Sample from mixture for each t
        # We'll use a simplified approach: deterministically assign samples
        # to components based on t value
        
        left_normal = Normal(self.left_mean, self.left_std)
        right_normal = Normal(right_mean, self.right_std)
        
        # Use t to decide which component
        use_left = torch.rand_like(t) < self.left_weight
        
        mask_rate = torch.where(
            use_left,
            left_normal.icdf(t.clamp(0.01, 0.99)),
            right_normal.icdf(t.clamp(0.01, 0.99))
        )
        
        mask_rate = mask_rate.clamp(self.eps, 1.0 - self.eps)
        alpha_t = 1 - mask_rate
        
        # Approximate derivative
        delta = 1e-4
        t_plus = (t + delta).clamp(self.eps, 1.0 - self.eps)
        mask_rate_plus = torch.where(
            use_left,
            left_normal.icdf(t_plus.clamp(0.01, 0.99)),
            right_normal.icdf(t_plus.clamp(0.01, 0.99))
        ).clamp(self.eps, 1.0 - self.eps)
        
        dalpha_t = -(mask_rate_plus - mask_rate) / delta
        
        return dalpha_t, alpha_t


# =============================================================================
# ELBO Derivative Softening
# =============================================================================

def soften_derivative(
    dalpha_t: torch.Tensor,
    power: float = 0.1,
) -> torch.Tensor:
    """Soften the ELBO derivative term by raising to a power.
    
    From the paper: Using the full derivative term (p=1.0) can lead to
    poor results with certain noise schedules. Softening (p < 1) or
    omitting (p=0) the derivative significantly improves performance,
    especially with bimodal Gaussian schedules.
    
    Args:
        dalpha_t: Time derivative of noise schedule, typically negative
        power: Softening power (0 = omit derivative, 1 = full derivative)
    
    Returns:
        softened: Softened derivative term
    """
    if power == 0.0:
        # Omit derivative entirely
        if torch.is_tensor(dalpha_t):
            return torch.ones_like(dalpha_t)
        return 1.0
    
    if power == 1.0:
        # No softening
        return dalpha_t
    
    # Apply softening: sign(d) * |d|^p
    if torch.is_tensor(dalpha_t):
        return torch.sign(dalpha_t) * torch.abs(dalpha_t) ** power
    else:
        sign = 1 if dalpha_t >= 0 else -1
        return sign * abs(dalpha_t) ** power


# =============================================================================
# Utility Functions
# =============================================================================

def create_frequency_masking_from_tokenizer(
    tokenizer,
    softening_power: float = 0.02,
    use_curriculum: bool = True,
) -> FrequencyInformedMasking:
    """Create frequency-informed masking using Zipf's law approximation.
    
    This is a convenience function when you don't have access to the
    actual token frequency statistics from the training data.
    
    Args:
        tokenizer: The tokenizer
        softening_power: Target power for softening weights
        use_curriculum: Whether to use curriculum learning
    
    Returns:
        FrequencyInformedMasking module
    """
    vocab_size = tokenizer.vocab_size
    frequency_weights = compute_frequency_weights_from_tokenizer(tokenizer)
    
    # Get special tokens
    special_tokens = []
    for attr in ['pad_token_id', 'bos_token_id', 'eos_token_id', 'unk_token_id', 'mask_token_id']:
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            special_tokens.append(token_id)
    
    mask_index = getattr(tokenizer, 'mask_token_id', None)
    
    return FrequencyInformedMasking(
        vocab_size=vocab_size,
        frequency_weights=frequency_weights,
        softening_power=softening_power,
        use_curriculum=use_curriculum,
        mask_index=mask_index,
        special_tokens=special_tokens,
    )
