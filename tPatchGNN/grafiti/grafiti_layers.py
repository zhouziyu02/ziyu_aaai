import torch
import torch.nn as nn
import torch.nn.functional as F
from grafiti.attention import MAB2 # Assuming this exists and is correct
 

def batch_flatten(x_list, mask):
    """
    Flatten a batch of time series based on a mask.

    Args:
        x_list (List[Tensor]): List of tensors with shape (B, T, C) or compatible.
                                 If shape is (B, T), it will be expanded internally.
        mask (Tensor): Mask tensor of shape (B, T, C) indicating elements to keep.

    Returns:
        List[Tensor]: List of flattened tensors with shape (B, K)
                      where K is the max number of True elements per batch instance.
    """
    # Ensure mask is boolean
    mask_bool = mask.bool()
    b, t, d = mask_bool.shape # Use mask shape as reference for T, D

    # Calculate K (max number of True elements per batch instance)
    observed_counts = mask_bool.view(b, -1).sum(dim=1)
    if observed_counts.numel() == 0 or observed_counts.max() == 0:
         k = 0
         # Return empty tensors or tensors of zeros with shape (b, 0)
         default_dtype = torch.float32 # Or determine from x_list if possible
         if x_list:
             if torch.is_tensor(x_list[0]):
                 default_dtype = x_list[0].dtype
         return [torch.empty((b, 0), device=mask.device, dtype=default_dtype) for _ in x_list]
    else:
        k = observed_counts.max().item()

    # Create indices for padding
    indices = torch.arange(k, device=mask.device).expand(b, k)
    mask_indices = indices < observed_counts.unsqueeze(1) # (B, K) boolean mask for valid positions

    y_padded_list = []
    for x in x_list:
        x_dtype = x.dtype # Get dtype for later use
        x_expanded = None # Initialize

        # Handle potential (B, T) shape by expanding (e.g., original obs_mask if passed)
        if x.ndim == 2 and x.shape[:2] == (b, t):
            x_expanded = x.unsqueeze(-1).expand(-1, -1, d) # Expand B,T -> B,T,D
        # Handle standard (B, T, D) float, int, or long tensors
        # ***** MODIFICATION HERE *****
        elif x.shape[:3] == (b, t, d) and x_dtype in [torch.float32, torch.float64, torch.int64, torch.long, torch.int32]: # Added int64, long, int32
             x_expanded = x
        # Handle (B, T, D) boolean masks explicitly
        elif x.shape[:3] == (b, t, d) and x_dtype == torch.bool:
             x_expanded = x
        # This check might be redundant now, but keep for safety or if mask is passed explicitly
        elif x.shape[:3] == (b, t, d) and x_dtype == torch.bool and torch.equal(x, mask_bool):
             x_expanded = x # Already correct shape and type
        else:
             # Raise error for unhandled shapes/types
             raise ValueError(f"Unexpected shape or type for tensor in x_list: {x.shape}, {x.dtype}")

        # Determine dtype for the padded tensor
        # Use original dtype, map bool to uint8 if needed for torch.zeros/full
        padded_dtype = x_dtype if x_dtype != torch.bool else torch.uint8

        # Flatten the (B, T, D) tensor
        x_flat_all = x_expanded.reshape(b, t * d)

        # Create padded tensor
        y_padded_ = torch.zeros((b, k), device=mask.device, dtype=padded_dtype)

        # Select and place values using the mask
        # Iterate through batch (simpler) or use advanced indexing (potentially faster but complex)
        for i in range(b):
            batch_mask_flat = mask_bool[i].reshape(-1) # T*D
            batch_values_flat = x_flat_all[i]         # T*D
            observed_values = batch_values_flat[batch_mask_flat] # num_observed_in_batch_i
            count = observed_counts[i].item()
            if count > 0: # Only assign if there are values
                 # Ensure dtype consistency during assignment
                 y_padded_[i, :count] = observed_values.to(padded_dtype)

        y_padded_list.append(y_padded_)

    return y_padded_list

# Keep reconstruct_y as it was, it works on the principle of target_mask (B, T, D)
# and a flattened prediction tensor (B, K'). The key is that the K' dimension
# in the flattened prediction tensor corresponds *only* to the True values
# in the original combined mask used for flattening.
def reconstruct_y(
    Y_target_mask: torch.Tensor, Y_flat_predictions: torch.Tensor, Y_flat_target_mask: torch.Tensor
) -> torch.Tensor:
    """
    Reconstructs the original tensor Y by placing predictions from Y_flat_predictions
    at locations specified by Y_target_mask. Uses Y_flat_target_mask to select
    the relevant predictions from Y_flat_predictions.

    Args:
        Y_target_mask: The original boolean mask of shape (B, T, D) indicating target locations.
        Y_flat_predictions: A tensor of shape (B, K'), containing predictions for *all*
                           points (observed and target) included in the initial flattening mask.
                           K' is the max number of observed+target points per batch item.
        Y_flat_target_mask: A tensor of shape (B, K') derived from flattening Y_target_mask.
                            Indicates which elements in Y_flat_predictions correspond to targets.

    Returns:
        Y_reconstructed: A tensor of shape (B, T, D) with predictions placed at target locations.
    """
    # Ensure masks are boolean
    Y_target_mask_bool = Y_target_mask.bool()
    Y_flat_target_mask_bool = Y_flat_target_mask.bool() # Shape (B, K')

    # Initialize output tensor
    Y_reconstructed = torch.zeros_like(Y_target_mask, dtype=Y_flat_predictions.dtype)

    # Get the indices (B, T, D) where targets are expected
    true_target_indices = torch.nonzero(Y_target_mask_bool, as_tuple=True)

    # Select the predictions that correspond to targets from the flattened prediction tensor
    # Y_flat_predictions has shape (B, K'), Y_flat_target_mask_bool has shape (B, K')
    # We need to select elements from Y_flat_predictions where Y_flat_target_mask_bool is True
    # The result should have a total number of elements equal to the number of True values in Y_target_mask_bool
    target_preds_flat = Y_flat_predictions[Y_flat_target_mask_bool] # This will be 1D: (total_num_targets,)

    # Place the selected predictions into the reconstructed tensor at the correct locations
    if target_preds_flat.numel() > 0: # Check if there are any targets
        Y_reconstructed[true_target_indices] = target_preds_flat

    return Y_reconstructed


def gather(x, inds):
    """
    Gather values from tensor based on indices.

    Args:
        x (Tensor): Tensor of shape (B, P, M)
        inds (Tensor): Indices of shape (B, K')

    Returns:
        Tensor: Gathered tensor of shape (B, K', M)
    """
    # Ensure inds is long type for gather
    inds = inds.long()
    # Handle empty inds case
    if inds.shape[1] == 0:
        return torch.empty((x.shape[0], 0, x.shape[-1]), device=x.device, dtype=x.dtype)
    return x.gather(1, inds.unsqueeze(-1).expand(-1, -1, x.shape[-1])) # Use expand instead of repeat


class grafiti_(nn.Module):
    """GraFITi model"""

    def __init__(
        self,
        dim: int = 41,
        nkernel: int = 128,
        n_layers: int = 3,
        attn_head: int = 4,
        device: str = "cuda",
    ):
        """initializing grafiti model

        Args:
            dim (int, optional): number of channels. Defaults to 41.
            nkernel (int, optional): latent dimension size. Defaults to 128.
            n_layers (int, optional): number of grafiti layers. Defaults to 3.
            attn_head (int, optional): number of attention heads. Defaults to 4.
            device (str, optional): "cpu" or "cuda. Defaults to "cuda".
        """
        super().__init__()
        self.nkernel = nkernel
        self.nheads = attn_head
        self.device = device # Note: device arg isn't used to move layers, usually done outside __init__
        self.n_layers = n_layers

        self.edge_init = nn.Linear(2, nkernel)
        self.chan_init = nn.Linear(dim, nkernel)
        self.time_init = nn.Linear(1, nkernel)

        self.channel_time_attn = nn.ModuleList(
            [
                MAB2(nkernel, 2 * nkernel, 2 * nkernel, nkernel, attn_head)
                for _ in range(n_layers)
            ]
        )
        self.time_channel_attn = nn.ModuleList(
            [
                MAB2(nkernel, 2 * nkernel, 2 * nkernel, nkernel, attn_head)
                for _ in range(n_layers)
            ]
        )
        self.edge_nn = nn.ModuleList(
            [nn.Linear(3 * nkernel, nkernel) for _ in range(n_layers)]
        )

        self.output = nn.Linear(3 * nkernel, 1)
        self.relu = nn.ReLU()

        # Move layers to device if specified (better practice: move model instance outside)
        # self.to(torch.device(device))

    def _one_hot_channels(
        self, batch_size: int, num_channels: int, device: torch.device
    ) -> torch.Tensor:
        """Creating onehot encoding of channel ids

        Args:
            batch_size (int): B
            num_channels (int): D
            device (torch.device): GPU or CPU

        Returns:
            torch.Tensor: onehot encoding of channels (B, D, D)
        """
        indices = torch.arange(num_channels, device=device).expand(
            batch_size, num_channels
        )
        # Ensure F.one_hot output matches expectation if chan_init expects float
        return F.one_hot(indices, num_classes=num_channels).float()


    def _build_indices(
        self,
        time_points: torch.Tensor,  # shape: (B, T) - Original shape
        num_channels: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Builds index tensors for time steps and channel IDs.

        Args:
            time_points (torch.Tensor): Input tensor with shape (B, T)
            num_channels (int): Number of channels (D)
            device (torch.device): CPU or GPU

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - t_inds (torch.Tensor): Time indices of shape (B, T, D)
                - c_inds (torch.Tensor): Channel indices of shape (B, T, D)
        """
        # time_points originally (B, T), no need to unsqueeze for shape info
        b, t = time_points.shape[0], time_points.shape[1]

        # Create time indices (B, T, D) -> Integer representing time step index
        t_inds = (
            torch.arange(t, device=device) # Shape (T,)
            .unsqueeze(0).unsqueeze(-1)    # Shape (1, T, 1)
            .expand(b, -1, num_channels)   # Shape (B, T, D)
        )

        # Create channel indices (B, T, D) -> Integer representing channel index
        c_inds = (
            torch.arange(num_channels, device=device) # Shape (D,)
            .unsqueeze(0).unsqueeze(0)                 # Shape (1, 1, D)
            .expand(b, t, -1)                          # Shape (B, T, D)
        )

        return t_inds, c_inds


    def _create_masks(
        self,
        mk_flat_bool: torch.Tensor, # flattened combined mask (B, K'), bool
        t_inds_flat: torch.Tensor,  # flattened time indices (B, K'), long/int
        c_inds_flat: torch.Tensor,  # flattened channel indices (B, K'), long/int
        t_len: int,                 # Original time dimension T
        c_onehot: torch.Tensor,     # onhot channel encoding; (B, D, D) -> Use shape info
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Creating masks for time and channel attentions in grafiti

        Args:
            mk_flat_bool (torch.Tensor): flattened combined mask; (B, K'), bool
            t_inds_flat (torch.Tensor): flattened time indices; (B, K'), long/int
            c_inds_flat (torch.Tensor): flattened channel indices: (B, K'), long/int
            t_len (int): Original sequence length T
            c_onehot (torch.Tensor): onhot channel encoding; (B, D, D)
            device (torch.Device): GPU or CPU

        Returns:
            tuple[torch.Tensor, torch.Tensor]
            t_mask: time attn mask (B, T, K') float
            c_mask: channel attn mask (B, D, K') float
        """
        b, num_channels = c_onehot.shape[:2]
        k_prime = mk_flat_bool.shape[1]

        # Handle empty K' case
        if k_prime == 0:
             return (
                 torch.empty((b, t_len, 0), device=device, dtype=torch.float),
                 torch.empty((b, num_channels, 0), device=device, dtype=torch.float)
             )

        # Channel Mask (B, D, K')
        # Compare each channel index (0 to D-1) with the flattened channel indices (c_inds_flat)
        chan_indices = torch.arange(num_channels, device=device)[None, :, None] # (1, D, 1)
        # c_inds_flat is (B, K'), needs shape (B, 1, K') for broadcasting
        c_match = (chan_indices == c_inds_flat[:, None, :]) # (B, D, K') bool
        # Mask should only be true where the original combined mask was true
        c_mask = c_match * mk_flat_bool[:, None, :] # (B, D, K') bool, applying combined mask

        # Time Mask (B, T, K')
        # Compare each time index (0 to T-1) with the flattened time indices (t_inds_flat)
        time_indices = torch.arange(t_len, device=device)[None, :, None] # (1, T, 1)
        # t_inds_flat is (B, K'), needs shape (B, 1, K') for broadcasting
        t_match = (time_indices == t_inds_flat[:, None, :]) # (B, T, K') bool
        # Mask should only be true where the original combined mask was true
        t_mask = t_match * mk_flat_bool[:, None, :] # (B, T, K') bool, applying combined mask

        # Convert to float for attention mechanisms if needed (often softmax handles bool implicitly, but float is safer)
        return t_mask.float(), c_mask.float()


    def _encode_features(
        self,
        u_raw: torch.Tensor,         # input edge feature (B, K', 2)
        t: torch.Tensor,             # time points (B, T, 1)
        c_onehot: torch.Tensor,      # channel node feature (B, D, D)
        mask_flat_bool: torch.Tensor # input combined mask, flattened (B, K'), bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encoding edge, time node and channel node features

        Args:
            u_raw (torch.Tensor): input edge feature (B, K', 2)
            t (torch.Tensor): time node feature (B, T, 1)
            c_onehot (torch.Tensor): channel node feature (B, D, D)
            mask_flat_bool (torch.Tensor): flattened combined mask (B, K'), bool

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            Encoded edge features (B, K', M),
            encoded time features (B, T, M),
            encoded channel features (B, D, M)

        """
        # Encode edges, apply mask to zero out padding embeddings
        u_encoded = self.relu(self.edge_init(u_raw)) * mask_flat_bool.unsqueeze(-1).float()  # (B, K', M)

        # Encode time (e.g., using sinusoidal embedding)
        t_encoded = torch.sin(self.time_init(t))  # (B, T, M)

        # Encode channels
        c_encoded = self.relu(self.chan_init(c_onehot))  # (B, D, M) - Assumes chan_init handles (B,D,D) input? Often expects (B,D,Din) -> check chan_init. If it expects (B, D, Din), c_onehot might need adjustment or chan_init should be changed. Assuming chan_init handles (B,D,D) -> (B,D,M).

        return u_encoded, t_encoded, c_encoded

    def forward(
        self,
        time_points: torch.Tensor, # (B, T)
        values: torch.Tensor,      # Observed values; Tensor (B, T, D)
        obs_mask: torch.Tensor,    # Observed time points mask; Tensor (B, T)
        target_mask: torch.Tensor, # Target values mask; Tensor (B, T, D)
    ) -> torch.Tensor:
        """GraFITi model

        Args:
            time_points: time_points have both observed and target times; Tensor (B, T)
            values: Observed values; Tensor (B, T, D). Assumed to be 0 where obs_mask is False.
            obs_mask: Observed time points mask; Tensor (B, T) boolean or float/int {0, 1}
            target_mask: Target values mask; Tensor (B, T, D) boolean or float/int {0, 1}

        Returns:
            yhat: Predictions; Tensor (B, T, D)
        """
        b, t_len, d = values.shape
        dev = values.device # Use device from input tensors

        # --- Input Preparation ---
        # Ensure masks are boolean
        obs_mask_bool = obs_mask.bool()       # (B, T)
        target_mask_bool = target_mask.bool() # (B, T, D)

        # Expand obs_mask to match target_mask dimensions: (B, T) -> (B, T, D)
        # A time point is observed context if obs_mask is True for that t, applies to all d
        obs_mask_expanded_bool = obs_mask_bool.unsqueeze(-1).expand(-1, -1, d) # (B, T, D)

        # Create combined mask for all points (observed context + targets) to be flattened
        combined_mask_bool = obs_mask_expanded_bool | target_mask_bool # (B, T, D)

        # Prepare values tensor for flattening: Use input 'values' only where observed, 0 otherwise.
        # Assumes input 'values' already respects obs_mask; if not, uncomment below
        # values_for_flattening = torch.zeros_like(values)
        # values_for_flattening[obs_mask_expanded_bool] = values[obs_mask_expanded_bool]
        # If 'values' is guaranteed to be 0 where not observed, we can use it directly:
        values_for_flattening = values # Use if values[b, t, d] = 0 if obs_mask[b, t] = False

        # Prepare time features (B, T, 1)
        t_feat = time_points.unsqueeze(-1)

        # Prepare channel features (one-hot) (B, D, D)
        c_onehot = self._one_hot_channels(b, d, device=dev)

        # Build integer indices for time and channels (B, T, D)
        t_inds, c_inds = self._build_indices(time_points, d, dev)

        # --- Flattening ---
        # Flatten all necessary tensors based on the combined mask
        # Input list for batch_flatten
        tensors_to_flatten = [
            t_inds,                  # Time indices (int)
            values_for_flattening,   # Values (observed or 0) (float)
            target_mask_bool,        # Target indicator (bool) -> will become uint8/float
            c_inds,                  # Channel indices (int)
            combined_mask_bool       # The combined mask itself (bool) -> needed? maybe just need count? Check usage. Let's keep it for now.
        ]
        flattened = batch_flatten(tensors_to_flatten, combined_mask_bool)
        # Check if flattening returned empty tensors (no observed/target points)
        if not flattened or flattened[0].shape[1] == 0:
             # Return zeros or appropriate shape if no points to process
             return torch.zeros_like(target_mask, dtype=values.dtype)

        t_inds_f, vals_f, tgt_mask_f, c_inds_f, combined_mask_f = flattened
        # t_inds_f, c_inds_f are (B, K'), long/int
        # vals_f is (B, K'), float (observed values or 0)
        # tgt_mask_f is (B, K'), bool/uint8 (1 if target, 0 otherwise)
        # combined_mask_f is (B, K'), bool/uint8 (1 if observed or target)

        # Convert flattened masks to boolean for indexing and float for calculations
        tgt_mask_f_bool = tgt_mask_f.bool()
        tgt_mask_f_float = tgt_mask_f.float() # Indicator for edge features
        combined_mask_f_bool = combined_mask_f.bool() # Mask for attention updates


        # --- Feature Preparation for Graph ---
        # Create edge features: [value, target_indicator] (B, K', 2)
        edge_input = torch.cat([vals_f.unsqueeze(-1), tgt_mask_f_float.unsqueeze(-1)], dim=-1)

        # Create attention masks for time and channel nodes (B, T, K') and (B, D, K')
        t_attn_mask, c_attn_mask = self._create_masks(
            combined_mask_f_bool, t_inds_f, c_inds_f, t_len, c_onehot, dev
        )

        # Encode initial node and edge features (B, K', M), (B, T, M), (B, D, M)
        edge_emb, t_emb, c_emb = self._encode_features(
            edge_input, t_feat, c_onehot, combined_mask_f_bool
        )

        # --- GraFITi Layers ---
        for i in range(self.n_layers):
            # Gather time/channel embeddings corresponding to the flattened points
            t_gathered = gather(t_emb, t_inds_f)  # (B, K', M)
            c_gathered = gather(c_emb, c_inds_f)  # (B, K', M)

            # Update channel embeddings based on time and edge info
            # MAB2(Q, K, V, output_dim, heads) -> MAB2(c_emb, cat(t_gath, edge), cat(t_gath, edge), nkernel, nheads)
            # Check if MAB2 expects mask for Q or K/V. Assuming mask applies to K/V attention scores.
            # c_attn_mask is (B, D, K') -> Seems like mask for K/V based on Q (channels)
            c_emb = self.channel_time_attn[i](
                c_emb, torch.cat([t_gathered, edge_emb], -1), c_attn_mask # Pass K'/V mask
            )  # Output: (B, D, M)

            # Update time embeddings based on channel and edge info
            # t_attn_mask is (B, T, K') -> Seems like mask for K/V based on Q (time)
            t_emb = self.time_channel_attn[i](
                t_emb, torch.cat([c_gathered, edge_emb], -1), t_attn_mask # Pass K'/V mask
            )  # Output: (B, T, M)

            # Prepare input for edge update MLP
            edge_update_input = torch.cat(
                [edge_emb, t_gathered, c_gathered], dim=-1
            )  # (B, K', 3*M)

            # Update edge embeddings (residual connection + MLP)
            # Apply combined mask again to zero out padding embeddings
            edge_emb = self.relu(edge_emb + self.edge_nn[i](edge_update_input))
            edge_emb = edge_emb * combined_mask_f_bool.unsqueeze(-1).float() # (B, K', M)

        # --- Output Generation ---
        # Gather final time/channel embeddings
        t_gathered_final = gather(t_emb, t_inds_f)  # (B, K', M)
        c_gathered_final = gather(c_emb, c_inds_f)  # (B, K', M)

        # Combine final embeddings for output MLP
        final_combined_emb = torch.cat([edge_emb, t_gathered_final, c_gathered_final], dim=-1) # (B, K', 3*M)

        # Get final predictions for all flattened points
        output_flat = self.output(final_combined_emb)  # (B, K', 1)

        # --- Reconstruction ---
        # Reconstruct the (B, T, D) tensor, placing predictions only at target locations
        # Pass the original target mask (B, T, D), the predictions (B, K', 1),
        # and the flattened target mask (B, K') to select relevant predictions.
        yhat = reconstruct_y(
            target_mask_bool, output_flat.squeeze(-1), tgt_mask_f_bool
        ) # Output: (B, T, D)

        return yhat