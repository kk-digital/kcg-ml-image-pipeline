import torch

from typing import List

from stable_diffusion.model.clip_text_embedder import CLIPTextEmbedder as SDCLIPTextEmbedder

from data_loader.clip_text_embedder_output import CLIPTextEmbedderOutput


class CLIPTextEmbedder(SDCLIPTextEmbedder):

    #TODO: add "inference_with_grad" 

    def forward(self, prompts: List[str]):

        with torch.no_grad():
            last_hidden_state, pooler_output, attention_mask = self.forward_return_all(prompts)

        results = list()
        for i in range(last_hidden_state.shape[0]):
            results.append(CLIPTextEmbedderOutput(
                model_name=self.model_name,
                prompts=prompts[i],
                embedding=last_hidden_state[i].detach().cpu().numpy(),
                pooler_output=pooler_output[i].detach().cpu().numpy(),
                attention_mask=attention_mask[i].detach().cpu().numpy()
            ))

        return results
