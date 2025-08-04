# Copyright 2024 IntelliGen-AI and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import numpy as np
from aurobind.openfold.model.embedders import (
    InputEmbedder,
    RecyclingEmbedder,
    TemplateEmbedder,
    MSAEmbedder,
)
from copy import deepcopy
from aurobind.openfold.model.pairformer import PairformerStack, MSAModuleStack
from aurobind.openfold.model.heads import AuxiliaryHeads


from aurobind.openfold.utils.tensor_utils import add


class BackboneTrunk(nn.Module):
    """
    IntelliFold Backbone

    Implements Algorithm 1 (part of) (but with training).
    """

    def __init__(self, config):
        """
        Args:
            config:
                A dict-like config object (like the one in config.py)
        """
        super(BackboneTrunk, self).__init__()

        self.globals = config.globals
        self.config = config.backbone
        self.recycling_iters = self.config.recycling_iters

        # Main trunk
        self.input_embedder = InputEmbedder(
            **self.config["input_embedder"],
        )

        self.recycling_embedder = RecyclingEmbedder(
            **self.config["recycling_embedder"],
        )

        if self.config.template_embedder.enabled:
            self.template_embedder = TemplateEmbedder(
                **self.config["template_embedder"],
            )

        self.msa_embedder = MSAEmbedder(
            **self.config["msa"]["msa_embedder"],
        )
        self.msa_stack = MSAModuleStack(
            **self.config["msa"]["msa_stack"],
        )

        self.pairformer = PairformerStack(
            **self.config["pairformer_stack"],
        )
        
        self.aux_heads = AuxiliaryHeads(
            self.config["heads"],
        )



    def iteration(self, feats, inits, prevs):
        # Primary output dictionary
        outputs = {}

        # This needs to be done manually for DeepSpeed's sake
        dtype = next(self.parameters()).dtype
        for k in feats:
            if feats[k].dtype == torch.float32:
                feats[k] = feats[k].to(dtype=dtype)

        batch_dims = feats["aatype"].shape[:-2]
        no_token = feats["aatype"].shape[-2]

        inplace_safe = False

        # Prep some features
        single_mask = feats["seq_mask"]
        pair_mask = single_mask[..., None] * single_mask[..., None, :]
        msa_mask = feats["msa_mask"]
        
        s_init, z_init, s_inputs = inits
        
        s_prev, z_prev = reversed([prevs.pop() for _ in range(2)])

        # Initialize the recycling embeddings, if needs be 
        if None in [s_prev, z_prev]:
            # [*, N, C_m]
            s_prev = s_init.new_zeros(
                (*batch_dims, no_token, self.config.input_embedder.c_s),
                requires_grad=False,
            )

            # [*, N, N, C_z]
            z_prev = z_init.new_zeros(
                (*batch_dims, no_token, no_token, self.config.input_embedder.c_z),
                requires_grad=False,
            )

        # s_prev_emb: [*, N, C_m]
        # z_prev_emb: [*, N, N, C_z]
        # -------------------------
        # START: Algo 1 line 8, 11
        # -------------------------
        s_prev_emb, z_prev_emb = self.recycling_embedder(
            s_prev,
            z_prev,
            inplace_safe=inplace_safe,
        )

        # [*, N, C_m]
        s = add(s_init, s_prev_emb, inplace=False)

        # [*, N, N, C_z]
        z = add(z_init, z_prev_emb, inplace=False)

        del s_prev, z_prev, s_prev_emb, z_prev_emb
        # -------------------------
        # END: Algo 1 line 8, 11
        # -------------------------
        
        # -------------------------
        # START: Algo 1 line 9
        # -------------------------
        # Embed the templates + merge with MSA/pair embeddings
        if self.config.template_embedder.enabled:
            
            template_embeds = self.template_embedder(
                feats,
                z,
                pair_mask.to(dtype=z.dtype),
                chunk_size=self.globals.chunk_size,
                use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                inplace_safe=inplace_safe,
                _mask_trans=self.config._mask_trans
            )

            # [*, N, N, C_z]
            z = add(z,
                    template_embeds,
                    inplace_safe,
                    )
        # -------------------------
        # END: Algo 1 line 9
        # -------------------------
        
        # -------------------------
        # START: Algo 1 line 10
        # -------------------------
        # Embed MSA features + merge with pairwise embeddings
        m ,msa_mask= self.msa_embedder(feats,s_inputs,msa_mask,inplace_safe=inplace_safe)

        # [*, N, N, C_z]
        m, z = self.msa_stack(
            m, z,
            msa_mask=msa_mask.to(dtype=m.dtype),
            pair_mask=pair_mask.to(dtype=z.dtype),
            chunk_size=self.globals.chunk_size,
            use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
            inplace_safe=inplace_safe,
            _mask_trans=self.config._mask_trans,
        )

        # -------------------------
        # END: Algo 1 line 10
        # -------------------------
        
        # -------------------------
        # START: Algo 1 line 12, 13
        # -------------------------
        
        # z: [*, N, N, C_z]
        # s: [*, N, C_s]          
        s, z = self.pairformer(
            s,
            z,
            single_mask=single_mask.to(dtype=s.dtype),
            pair_mask=pair_mask.to(dtype=z.dtype),
            chunk_size=self.globals.chunk_size,
            use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
            inplace_safe=inplace_safe,
            _mask_trans=self.config._mask_trans,
        )

        outputs["z"] = z
        outputs["s"] = s
        outputs["s_inputs"] = s_inputs

        del z, s, m

        # [*, N, C_m]
        s_prev = outputs["s"]

        # [*, N, N, C_z]
        z_prev = outputs["z"]


        return outputs, s_prev, z_prev
    
        # -------------------------
        # END: Algo 1 line 12, 13
        # -------------------------

    def forward(self, feats):
        """
        Args:
            feats (dict): Dictionary of features, as outlined in Algorithm 5.
        Returns:
            Output of the forward pass.
        """
        inplace_safe = not (self.training or torch.is_grad_enabled())
        inplace_safe = False
        # Initialize recycling embeddings
        s_prev, z_prev = None, None
        prevs = [s_prev, z_prev]

        is_grad_enabled = torch.is_grad_enabled()

        # Main recycling loop
        num_iters = self.recycling_iters + 1
        
        # Initialize the single and pair representations, along with s_inputs
        # s: [*, N, C_s]
        # z: [*, N, N, C_z]
        # s_inputs: [*, N, C_s]
        # -------------------------
        # START: Algo 1 line 1-5
        # -------------------------
        s_init, z_init, s_inputs = self.input_embedder(
            feats,
            chunk_size=self.globals.chunk_size,
            use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
            inplace_safe=inplace_safe,)
        inits = [s_init, z_init, s_inputs]
        # -------------------------
        # END: Algo 1 line 1-5
        # -------------------------

        for cycle_no in range(num_iters):
            # Select the features for the current recycling cycle

            is_final_iter = cycle_no == (num_iters - 1)
            # Run the next iteration of the model
            outputs, s_prev, z_prev = self.iteration(
                feats,
                inits,
                prevs,
            )

            if not is_final_iter:
                del outputs
                prevs = [s_prev, z_prev]
                del s_prev, z_prev
            else:
                break

        # Run auxiliary heads

        # outputs.update(self.aux_heads(outputs))
        outputs.update(self.aux_heads(outputs,feats))

        return outputs
