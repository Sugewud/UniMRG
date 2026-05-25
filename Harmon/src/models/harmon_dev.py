import torch
import torch.nn.functional as F
from torch.nn.modules.module import T
from mmengine.model import BaseModel
from torch.autograd.function import Function
from mmengine.logging import print_log
from xtuner.model.utils import guess_load_checkpoint
from xtuner.utils import IMAGE_TOKEN_INDEX
from .harmon import Harmon


class _ScaleGradient(Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


class HarmonDev(Harmon, BaseModel):
    def __init__(self,
                 grad_scale=0.1,
                 loss_weights={'image2text': 1.0, 'text2image': 1.0, 'recon': 1.0, 'edit': 1.0},
                 pretrained_pth=None,
                 freeze_llm=False,
                 freeze_mar_encoder=False,
                 freeze_mar_decoder=False,
                 freeze_proj_in=False,
                 freeze_proj_out=False,
                 gradient_checkpointing=True,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.grad_scale = grad_scale
        self.loss_weights = loss_weights
        self.debuged = False

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            info = self.load_state_dict(pretrained_state_dict, strict=False)
            print_log(f'Load pretrained weight from {pretrained_pth}')

        # Freeze specific modules
        if freeze_llm:
            self.llm.requires_grad_(False)
            print_log('Frozen LLM')
        
        if freeze_mar_encoder:
            # Freeze MAR encoder blocks and related components
            for param in self.mar.encoder_blocks.parameters():
                param.requires_grad = False
            self.mar.z_proj.requires_grad_(False)
            self.mar.z_proj_ln.requires_grad_(False)
            self.mar.encoder_pos_embed_learned.requires_grad = False
            self.mar.encoder_norm.requires_grad_(False)
            print_log('Frozen MAR Encoder')
        
        if freeze_mar_decoder:
            # Freeze MAR decoder blocks and related components
            for param in self.mar.decoder_blocks.parameters():
                param.requires_grad = False
            self.mar.decoder_embed.requires_grad_(False)
            self.mar.mask_token.requires_grad = False
            self.mar.decoder_pos_embed_learned.requires_grad = False
            self.mar.decoder_norm.requires_grad_(False)
            self.mar.diffusion_pos_embed_learned.requires_grad = False
            self.mar.diffloss.requires_grad_(False)
            print_log('Frozen MAR Decoder')
        
        if freeze_proj_in:
            self.proj_in.requires_grad_(False)
            print_log('Frozen proj_in')
        
        if freeze_proj_out:
            self.proj_out.requires_grad_(False)
            print_log('Frozen proj_out')

        # gradient checkpointing
        if gradient_checkpointing:
            self.gradient_checkpointing_enable()
        else:
            self.gradient_checkpointing_disable()

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.mar.gradient_checkpointing_disable()

    def gradient_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        self.mar.gradient_checkpointing_enable()

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        state_dict = {k: v for k, v in state_dict.items()
                      if 'vae.' not in k}

        return state_dict

    def train(self: T, mode: bool = True) -> T:
        super().train(mode=mode)
        self.vae.train(mode=False)
        return self

    def text2image_loss(self, data_dict):
        x = data_dict['pixel_values'].to(dtype=self.dtype, device=self.device)
        x = self.encode(x)   # b m n c
        b, m, n, _ = x.shape
        gt_latents = x.clone().detach().view(b, m*n, -1)

        orders = self.mar.sample_orders(bsz=b, seq_len=m*n)
        mask = self.mar.random_masking(x.flatten(1, 2), orders)

        input_ids = data_dict['input_ids'].to(self.device)
        attention_mask = data_dict['attention_mask'].to(self.device)
        x_enc = self.forward_mae_encoder(x, mask, input_ids=input_ids,
                                         attention_mask=attention_mask)
        z = self.mar.forward_mae_decoder(x_enc, mask, image_shape=(m, n))

        loss = self.mar.forward_loss(z=z, target=gt_latents, mask=mask)

        return loss

    def image2text_loss(self, data_dict):
        input_ids = data_dict['input_ids'].to(self.device)
        attention_mask = data_dict['attention_mask'].to(self.device)
        labels = data_dict['labels'].to(self.device)
        pixel_values = data_dict.get('pixel_values', None)
        if pixel_values is None:
            assert False
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            _, z_null = self.extract_visual_feature(
                torch.zeros(1, 16, 16, self.token_embed_dim,
                            dtype=self.dtype, device=self.device)
            )
            loss_null = z_null.mean() * 0.0
            print(f"No image found in this batch!", flush=True)
        else:
            x = pixel_values.to(dtype=self.dtype, device=self.device)
            x = self.encode(x)  # b m n c
            _, z_enc = self.extract_visual_feature(x)

            if self.grad_scale is not None:
                z_enc = _ScaleGradient.apply(z_enc, self.grad_scale)

            for i in range(input_ids.shape[0]):
                assert input_ids[i, 3] == IMAGE_TOKEN_INDEX
            input_ids = torch.cat(
                (
                    input_ids[:, :3],
                    IMAGE_TOKEN_INDEX * torch.ones(
                        input_ids.shape[0], 1088, dtype=torch.long, device=input_ids.device),
                    input_ids[:, 4:],
                ), dim=1
            )
            attention_mask = torch.cat(
                (
                    attention_mask[:, :3],
                    torch.ones(
                        attention_mask.shape[0], 1088, dtype=torch.long, device=attention_mask.device),
                    attention_mask[:, 4:],
                ), dim=1
            )
            labels = torch.cat(
                (
                    labels[:, :3],
                    labels[0, 3] * torch.ones(
                        labels.shape[0], 1088 , dtype=torch.long, device=labels.device),
                    labels[:, 4:],
                ), dim=1
            )
            inputs_embeds = z_enc.new_zeros(*input_ids.shape, self.llm.config.hidden_size)
            inputs_embeds[input_ids == IMAGE_TOKEN_INDEX] = z_enc.flatten(0, 1)
            inputs_embeds[input_ids != IMAGE_TOKEN_INDEX] = self.llm.get_input_embeddings()(
                input_ids[input_ids != IMAGE_TOKEN_INDEX])
            loss_null = 0.0

        output = self.llm_model(inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask,
                                return_dict=True)

        last_hidden_state = output.last_hidden_state[:, :-1]
        labels = labels[:, 1:]
        last_hidden_state = last_hidden_state[labels >= 0]
        labels = labels[labels >= 0]
        logits = self.llm.get_output_embeddings()(last_hidden_state)

        loss_i2t = F.cross_entropy(input=logits, target=labels)

        return loss_i2t + loss_null

    def recon_loss(self, data_dict):

        x = data_dict['pixel_values'].to(dtype=self.dtype, device=self.device)
        x = self.encode(x)   # b m n c
        b, m, n, _ = x.shape
        gt_latents = x.clone().detach().view(b, m*n, -1)
        
        input_ids = data_dict['input_ids'].to(self.device)
        attention_mask = data_dict['attention_mask'].to(self.device)
        
        _, z_enc = self.extract_visual_feature(x)
        
        if self.grad_scale is not None:
            z_enc = _ScaleGradient.apply(z_enc, self.grad_scale)
        
        for i in range(input_ids.shape[0]):
            assert input_ids[i, 3] == IMAGE_TOKEN_INDEX

        input_ids = torch.cat(
            (
                input_ids[:, :3],
                IMAGE_TOKEN_INDEX * torch.ones(
                    input_ids.shape[0], 1088, dtype=torch.long, device=input_ids.device),
                input_ids[:, 4:],
            ), dim=1
        )
        attention_mask = torch.cat(
            (
                attention_mask[:, :3],
                torch.ones(
                    attention_mask.shape[0], 1088, dtype=torch.long, device=attention_mask.device),
                attention_mask[:, 4:],
            ), dim=1
        )
        inputs_embeds = z_enc.new_zeros(*input_ids.shape, self.llm.config.hidden_size)
        inputs_embeds[input_ids == IMAGE_TOKEN_INDEX] = z_enc.flatten(0, 1)
        inputs_embeds[input_ids != IMAGE_TOKEN_INDEX] = self.llm.get_input_embeddings()(
            input_ids[input_ids != IMAGE_TOKEN_INDEX])

        orders = self.mar.sample_orders(bsz=b, seq_len=m*n)
        mask = self.mar.random_masking(x.flatten(1, 2), orders)
        
        x_enc = self.forward_mae_encoder(
            x, 
            mask, 
            inputs_embeds=inputs_embeds,
            input_ids=None,
            attention_mask=attention_mask
        )
        
        z = self.mar.forward_mae_decoder(x_enc, mask, image_shape=(m, n))
        loss = self.mar.forward_loss(z=z, target=gt_latents, mask=mask)

        return loss

    def edit_loss(self, data_dict):
        """Image editing loss.
        
        Unlike reconstruction which uses the same image as input/output,
        editing uses source_image as condition and edited_image as target.
        """
        # Encode source image (condition)
        source_x = data_dict['source_pixel_values'].to(dtype=self.dtype, device=self.device)
        source_x = self.encode(source_x)  # b m n c
        
        # Encode target edited image
        target_x = data_dict['pixel_values'].to(dtype=self.dtype, device=self.device)
        target_x = self.encode(target_x)   # b m n c
        b, m, n, _ = target_x.shape
        gt_latents = target_x.clone().detach().view(b, m*n, -1)
        
        input_ids = data_dict['input_ids'].to(self.device)
        attention_mask = data_dict['attention_mask'].to(self.device)
        
        # Extract visual features from source image (as condition)
        _, z_enc = self.extract_visual_feature(source_x)
        
        if self.grad_scale is not None:
            z_enc = _ScaleGradient.apply(z_enc, self.grad_scale)
        
        for i in range(input_ids.shape[0]):
            assert input_ids[i, 3] == IMAGE_TOKEN_INDEX

        input_ids = torch.cat(
            (
                input_ids[:, :3],
                IMAGE_TOKEN_INDEX * torch.ones(
                    input_ids.shape[0], 1088, dtype=torch.long, device=input_ids.device),
                input_ids[:, 4:],
            ), dim=1
        )
        attention_mask = torch.cat(
            (
                attention_mask[:, :3],
                torch.ones(
                    attention_mask.shape[0], 1088, dtype=torch.long, device=attention_mask.device),
                attention_mask[:, 4:],
            ), dim=1
        )
        inputs_embeds = z_enc.new_zeros(*input_ids.shape, self.llm.config.hidden_size)
        inputs_embeds[input_ids == IMAGE_TOKEN_INDEX] = z_enc.flatten(0, 1)
        inputs_embeds[input_ids != IMAGE_TOKEN_INDEX] = self.llm.get_input_embeddings()(
            input_ids[input_ids != IMAGE_TOKEN_INDEX])

        # Generate target edited image
        orders = self.mar.sample_orders(bsz=b, seq_len=m*n)
        mask = self.mar.random_masking(target_x.flatten(1, 2), orders)
        
        x_enc = self.forward_mae_encoder(
            target_x, 
            mask, 
            inputs_embeds=inputs_embeds,
            input_ids=None,
            attention_mask=attention_mask
        )
        
        z = self.mar.forward_mae_decoder(x_enc, mask, image_shape=(m, n))
        loss = self.mar.forward_loss(z=z, target=gt_latents, mask=mask)

        return loss

    def forward(self, data, data_samples=None, mode='loss'):
        if mode == 'loss':
            return self.compute_loss(data_dict=data)
        else:
            raise NotImplementedError

    def joint_loss(self, data_dict):
        """Joint loss for SFT and Depth estimation in a single step.
        
        This method manually batches inputs for SFT and Depth tasks to avoid
        calling shared modules (like Visual Encoder and LLM) multiple times
        in a single forward pass, which causes "Gradient computed twice" errors in DeepSpeed.
        """
        # --- 1. Prepare Inputs ---
        
        # SFT Data
        sft_input_ids = data_dict['sft_input_ids'].to(self.device)
        sft_labels = data_dict['sft_labels'].to(self.device)
        sft_attention_mask = data_dict['sft_attention_mask'].to(self.device)
        sft_pixel_values = data_dict['sft_pixel_values'].to(dtype=self.dtype, device=self.device)
        
        # Depth Data
        depth_source_pixel_values = data_dict['depth_source_pixel_values'].to(dtype=self.dtype, device=self.device)
        depth_target_pixel_values = data_dict['depth_pixel_values'].to(dtype=self.dtype, device=self.device)
        depth_input_ids = data_dict['depth_input_ids'].to(self.device)
        depth_attention_mask = data_dict['depth_attention_mask'].to(self.device)
        
        b_sft = sft_input_ids.shape[0]
        b_depth = depth_input_ids.shape[0]
        
        # --- 2. Shared VAE Encoding (Safe to call multiple times if frozen, but efficient to batch) ---
        # Concatenate all images for encoding: [SFT, Depth_Source, Depth_Target]
        all_images = torch.cat([sft_pixel_values, depth_source_pixel_values, depth_target_pixel_values], dim=0)
        all_latents = self.encode(all_images)
        
        sft_latents, depth_source_latents, depth_target_latents = torch.split(all_latents, [b_sft, b_depth, b_depth], dim=0)
        
        # --- 3. Shared Visual Encoder (MUST batch this if trainable) ---
        # Concatenate latents for visual feature extraction: [SFT, Depth_Source]
        # Note: Depth_Target also needs visual features inside forward_mae_encoder, but let's see.
        # In forward_mae_encoder(target_x), it calls extract_visual_feature(target_x).
        # So we need to extract features for ALL 3 sets of latents.
        
        all_vis_latents = torch.cat([sft_latents, depth_source_latents, depth_target_latents], dim=0)
        
        # We also need to generate masks for Depth_Target (for MAR)
        # SFT and Depth_Source don't use masks (mask=None/zeros) for feature extraction usually
        # But forward_mae_encoder passes a mask.
        
        # Generate masks for Depth_Target
        orders = self.mar.sample_orders(bsz=b_depth, seq_len=depth_target_latents.shape[1]*depth_target_latents.shape[2])
        depth_mask = self.mar.random_masking(depth_target_latents.flatten(1, 2), orders)
        
        # For SFT and Source, mask is zeros (no masking)
        # We need to create a combined mask for batching
        # Mask shape for one image: [b, mn]
        mask_zeros = torch.zeros(b_sft + b_depth, depth_mask.shape[1], device=self.device, dtype=depth_mask.dtype)
        all_masks = torch.cat([mask_zeros, depth_mask], dim=0)
        
        # Extract Visual Features (One pass)
        # This calls mar.forward_mae_encoder -> BUT wait, forward_mae_encoder calls llm_model!
        # Harmon.extract_visual_feature calls self.mar.forward_mae_encoder. 
        # Is self.mar.forward_mae_encoder the same as self.forward_mae_encoder?
        # NO. self.mar is the MAR module; self.forward_mae_encoder is a method on the Harmon class.
        # Harmon.extract_visual_feature calls self.mar.forward_mae_encoder.
        # Harmon.forward_mae_encoder calls extract_visual_feature AND llm_model.
        
        # So extract_visual_feature runs the Vision Transformer.
        # We run this ONCE.
        
        all_x_enc, all_z_enc = self.extract_visual_feature(all_vis_latents, mask=all_masks)
        
        # Split results
        # z_enc is the projected features used as inputs to LLM
        # x_enc is the features from Vision Encoder (before projection?) No, extract_visual_feature returns proj_in(x_enc).
        # Actually: x_enc from mar.forward_mae_encoder, z_enc = proj_in(x_enc).
        # We need z_enc for SFT and Depth Source.
        # We need x_enc for Depth Target (residual connection).
        
        z_enc_sft, z_enc_source, z_enc_target = torch.split(all_z_enc, [b_sft, b_depth, b_depth], dim=0)
        x_enc_target = torch.split(all_x_enc, [b_sft, b_depth, b_depth], dim=0)[2]
        
        # Apply scale gradient if needed
        if self.grad_scale is not None:
            z_enc_sft = _ScaleGradient.apply(z_enc_sft, self.grad_scale)
            z_enc_source = _ScaleGradient.apply(z_enc_source, self.grad_scale)
            # Target features usually don't get scaled? edit_loss doesn't scale target features explicitly, 
            # but forward_mae_encoder calls extract_visual_feature which returns z_enc.
            # In forward_mae_encoder, z_enc is used as input to LLM.
            # Does it scale? In edit_loss code:
            # x_enc = self.forward_mae_encoder(...)
            # Inside forward_mae_encoder: x_enc, z_enc = self.extract_visual_feature(..., detach=detach)
            # It doesn't seem to apply scale gradient inside forward_mae_encoder.
            # So we apply it only for SFT and Source (Condition).
        
        # --- 4. Prepare LLM Inputs ---
        
        # Get actual visual token length from encoder output
        vis_token_len = z_enc_sft.shape[1]

        # A. SFT Inputs construction (Logic from image2text_loss)
        for i in range(sft_input_ids.shape[0]):
            assert sft_input_ids[i, 3] == IMAGE_TOKEN_INDEX
            
        # Insert Image Tokens placeholders
        sft_input_ids_exp = torch.cat((
            sft_input_ids[:, :3],
            IMAGE_TOKEN_INDEX * torch.ones(b_sft, vis_token_len, dtype=torch.long, device=self.device),
            sft_input_ids[:, 4:]), dim=1)
            
        sft_attention_mask_exp = torch.cat((
            sft_attention_mask[:, :3],
            torch.ones(b_sft, vis_token_len, dtype=torch.long, device=self.device),
            sft_attention_mask[:, 4:]), dim=1)
            
        # SFT Labels (for loss later)
        sft_labels_exp = torch.cat((
            sft_labels[:, :3],
            sft_labels[0, 3] * torch.ones(b_sft, vis_token_len, dtype=torch.long, device=self.device),
            sft_labels[:, 4:]), dim=1)
            
        # Embeddings
        sft_inputs_embeds = z_enc_sft.new_zeros(*sft_input_ids_exp.shape, self.llm.config.hidden_size)
        sft_inputs_embeds[sft_input_ids_exp == IMAGE_TOKEN_INDEX] = z_enc_sft.flatten(0, 1)
        mask_text = sft_input_ids_exp != IMAGE_TOKEN_INDEX
        sft_inputs_embeds[mask_text] = self.llm.get_input_embeddings()(sft_input_ids_exp[mask_text])
        
        # B. Depth Inputs construction (Logic from edit_loss -> forward_mae_encoder)
        # First prepare the "Condition" embeddings (Source Image + Prompt)
        for i in range(depth_input_ids.shape[0]):
            assert depth_input_ids[i, 3] == IMAGE_TOKEN_INDEX
            
        depth_input_ids_exp = torch.cat((
            depth_input_ids[:, :3],
            IMAGE_TOKEN_INDEX * torch.ones(b_depth, vis_token_len, dtype=torch.long, device=self.device),
            depth_input_ids[:, 4:]), dim=1)
            
        depth_attention_mask_exp = torch.cat((
            depth_attention_mask[:, :3],
            torch.ones(b_depth, vis_token_len, dtype=torch.long, device=self.device),
            depth_attention_mask[:, 4:]), dim=1)
            
        # Condition Embeddings (Source + Prompt)
        depth_cond_embeds = z_enc_source.new_zeros(*depth_input_ids_exp.shape, self.llm.config.hidden_size)
        depth_cond_embeds[depth_input_ids_exp == IMAGE_TOKEN_INDEX] = z_enc_source.flatten(0, 1)
        mask_cond_text = depth_input_ids_exp != IMAGE_TOKEN_INDEX
        depth_cond_embeds[mask_cond_text] = self.llm.get_input_embeddings()(depth_input_ids_exp[mask_cond_text])
        
        # Now combine Condition with Target (Logic from prepare_forward_input)
        # inputs_embeds = torch.cat([inputs_embeds, x], dim=1) where x is z_enc_target
        depth_full_embeds = torch.cat([depth_cond_embeds, z_enc_target], dim=1)
        
        # Prepare Mask for Depth (Logic from prepare_forward_input)
        # attention_mask = torch.cat([attention_mask, attention_mask.new_ones(b, l)], dim=1)
        l_target = z_enc_target.shape[1]
        depth_full_mask = torch.cat([
            depth_attention_mask_exp, 
            depth_attention_mask_exp.new_ones(b_depth, l_target)
        ], dim=1)
        
        # --- 5. Batch LLM Forward ---
        # We need to concat sft_inputs_embeds and depth_full_embeds.
        # They might have different lengths. We need to pad.
        
        len_sft = sft_inputs_embeds.shape[1]
        len_depth = depth_full_embeds.shape[1]
        max_len = max(len_sft, len_depth)
        
        # Pad SFT
        if len_sft < max_len:
            pad_len = max_len - len_sft
            pad_embeds = torch.zeros(b_sft, pad_len, sft_inputs_embeds.shape[2], device=self.device, dtype=self.dtype)
            sft_inputs_embeds = torch.cat([sft_inputs_embeds, pad_embeds], dim=1)
            pad_mask = torch.zeros(b_sft, pad_len, device=self.device, dtype=sft_attention_mask_exp.dtype)
            sft_attention_mask_exp = torch.cat([sft_attention_mask_exp, pad_mask], dim=1)
            
        # Pad Depth
        if len_depth < max_len:
            pad_len = max_len - len_depth
            pad_embeds = torch.zeros(b_depth, pad_len, depth_full_embeds.shape[2], device=self.device, dtype=self.dtype)
            depth_full_embeds = torch.cat([depth_full_embeds, pad_embeds], dim=1)
            pad_mask = torch.zeros(b_depth, pad_len, device=self.device, dtype=depth_full_mask.dtype)
            depth_full_mask = torch.cat([depth_full_mask, pad_mask], dim=1)
            
        # Concat
        joint_inputs_embeds = torch.cat([sft_inputs_embeds, depth_full_embeds], dim=0)
        joint_attention_mask = torch.cat([sft_attention_mask_exp, depth_full_mask], dim=0)
        
        # Forward LLM (ONE PASS!)
        output = self.llm_model(inputs_embeds=joint_inputs_embeds,
                                attention_mask=joint_attention_mask,
                                return_dict=True)
        
        last_hidden_state = output.last_hidden_state
        
        # --- 6. Compute Losses ---
        
        # A. SFT Loss
        # Slice out SFT part
        sft_hidden = last_hidden_state[:b_sft, :len_sft] # Remove padding
        
        # Logic from image2text_loss
        # shift labels and hidden states
        sft_hidden = sft_hidden[:, :-1]
        sft_labels_shifted = sft_labels_exp[:, 1:]
        
        # Flatten and filter
        # Note: sft_labels_exp was not padded in Step 5, so we use original shape logic
        # BUT sft_hidden is taken from padded batch. We sliced to len_sft.
        # However, sft_labels_exp matches len_sft (the length before padding).
        
        sft_hidden_flat = sft_hidden[sft_labels_shifted >= 0]
        sft_labels_flat = sft_labels_shifted[sft_labels_shifted >= 0]
        
        sft_logits = self.llm.get_output_embeddings()(sft_hidden_flat)
        loss_sft = F.cross_entropy(input=sft_logits, target=sft_labels_flat)
        
        # B. Depth Loss
        # Slice out Depth part
        depth_hidden = last_hidden_state[b_sft:, :len_depth]
        
        # Logic from forward_mae_encoder: z_llm = output.last_hidden_state[:, -z_enc.shape[1]:]
        # z_enc here refers to z_enc_target (length 256 usually)
        # depth_hidden contains [Condition, Target]. We want the part corresponding to Target.
        # Target length is l_target
        z_llm_depth = depth_hidden[:, -l_target:]
        
        # Logic: move buffers back (from forward_mae_encoder)
        z_llm_depth = torch.cat([
            z_llm_depth[:, -self.mar.buffer_size:],
            z_llm_depth[:, :-self.mar.buffer_size]], dim=1)
            
        # Residual learning
        x_enc_final = x_enc_target + self.proj_out(z_llm_depth)
        
        # Decoder and Loss
        gt_latents = depth_target_latents.clone().detach().view(b_depth, -1, depth_target_latents.shape[-1]) # Flatten m*n
        z_dec = self.mar.forward_mae_decoder(x_enc_final, depth_mask, image_shape=(16, 16)) # Assuming 16x16 patches?
        # Need to get m, n from latents shape.
        # all_latents shape: [b, m, n, c]
        m, n = depth_target_latents.shape[1], depth_target_latents.shape[2]
        
        loss_depth = self.mar.forward_loss(z=z_dec, target=gt_latents, mask=depth_mask)
        
        return loss_sft, loss_depth

    def compute_loss(self, data_dict):
        losses = {}
        for data_type, batch_data in data_dict.items():
            if 'text2image' in data_type:
                loss = self.text2image_loss(batch_data)
                losses[f'loss_{data_type}'] = loss * self.loss_weights[data_type]
            elif 'image2text' in data_type:
                loss = self.image2text_loss(batch_data)
                losses[f'loss_{data_type}'] = loss * self.loss_weights[data_type]
            elif 'recon' in data_type:
                loss = self.recon_loss(batch_data)
                losses[f'loss_{data_type}'] = loss * self.loss_weights[data_type]
            elif 'depth' in data_type:
                # Depth reconstruction (source image -> target depth map)
                loss = self.edit_loss(batch_data)
                losses[f'loss_{data_type}'] = loss * self.loss_weights.get(data_type, 1.0)
            elif 'sketch' in data_type:
                # Sketch generation (source image -> target sketch)
                loss = self.edit_loss(batch_data)
                losses[f'loss_{data_type}'] = loss * self.loss_weights.get(data_type, 1.0)
            elif 'mask' in data_type:
                # Mask segmentation (source image -> target mask)
                loss = self.edit_loss(batch_data)
                losses[f'loss_{data_type}'] = loss * self.loss_weights.get(data_type, 1.0)
            elif 'joint' in data_type:
                loss_sft, loss_depth = self.joint_loss(batch_data)
                # Weight joint loss parts using individual weights if available, or just sum
                w_sft = self.loss_weights.get('image2text', 1.0)
                w_depth = self.loss_weights.get('edit', 1.0)
                # Use 'joint' weight as global scaler for this branch
                w_joint = self.loss_weights.get('joint', 1.0)
                
                losses[f'loss_{data_type}_sft'] = loss_sft * w_sft * w_joint
                losses[f'loss_{data_type}_depth'] = loss_depth * w_depth * w_joint
            else:
                raise NotImplementedError(f"Unknown data type: {data_type}")
        return losses
