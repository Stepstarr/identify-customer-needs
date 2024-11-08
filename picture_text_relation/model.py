from typing import List
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoTokenizer, AutoModel, AutoConfig
from dataset import MyPair
import constants

# constants for model
CLS_POS = 0
IMAGE_SIZE = 224
VISUAL_LENGTH = (IMAGE_SIZE // 32) ** 2


def resnet_encode(model, x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = x.view(x.size()[0], x.size()[1], -1)
    x = x.transpose(1, 2)

    return x


class MyModel(nn.Module):
    def __init__(
            self,
            encoder_t: PreTrainedModel,
            encoder_v: nn.Module,
            tokenizer: PreTrainedTokenizer,
            num_classes: int,
    ):
        super().__init__()
        self.encoder_t = encoder_t
        self.encoder_v = encoder_v
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.encoder_t = self.encoder_t.to(self.device)
        self.encoder_v = self.encoder_v.to(self.device)
        
        self.proj = nn.Linear(encoder_v.fc.in_features, encoder_t.config.hidden_size)
        self.aux_head = nn.Linear(encoder_t.config.hidden_size, num_classes)
        
        self.proj = self.proj.to(self.device)
        self.aux_head = self.aux_head.to(self.device)

    @classmethod
    def from_pretrained(cls, args):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        models_path = 'picture_text_relation/resources'

        encoder_t_path = f'{models_path}/{args.encoder_t}'
        print(encoder_t_path)
        tokenizer = AutoTokenizer.from_pretrained(encoder_t_path, local_files_only=True)
        encoder_t = AutoModel.from_pretrained(encoder_t_path, local_files_only=True)
        config = AutoConfig.from_pretrained(encoder_t_path, local_files_only=True)
        hid_dim_t = config.hidden_size

        encoder_v = getattr(torchvision.models, args.encoder_v)(pretrained=False)
        
        encoder_v.load_state_dict(torch.load(f'{models_path}/cnn/{args.encoder_v}.pth', map_location=device))
        hid_dim_v = encoder_v.fc.in_features

        return cls(
            encoder_t=encoder_t,
            encoder_v=encoder_v,
            tokenizer=tokenizer,
            num_classes=2,  # 假设有两个类别
        )

    def _bert_forward_with_image(self, inputs, image_explan_inputs, pairs):
        images = [pair.image for pair in pairs]
        textual_embeds = self.encoder_t.embeddings.word_embeddings(inputs.input_ids)
        image_explan_embeds = self.encoder_t.embeddings.word_embeddings(image_explan_inputs.input_ids)
        
        visual_embeds = torch.stack([image.data for image in images]).to(self.device)
        visual_embeds = resnet_encode(self.encoder_v, visual_embeds)
        visual_embeds = self.proj(visual_embeds)
        
        inputs_embeds = torch.concat((textual_embeds, image_explan_embeds, visual_embeds), dim=1)

        batch_size = visual_embeds.size()[0]
        visual_length = visual_embeds.size()[1]

        attention_mask = torch.cat((inputs.attention_mask, image_explan_inputs.attention_mask), dim=1)
        visual_mask = torch.ones((batch_size, visual_length), dtype=attention_mask.dtype, device=self.device)
        attention_mask = torch.cat((attention_mask, visual_mask), dim=1)

        token_type_ids = torch.cat((inputs.token_type_ids, image_explan_inputs.token_type_ids), dim=1)
        visual_type_ids = torch.ones((batch_size, visual_length), dtype=token_type_ids.dtype, device=self.device)
        token_type_ids = torch.cat((token_type_ids, visual_type_ids), dim=1)

        return self.encoder_t(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

    def itr_forward(self, pairs: List[MyPair]):
        text_batch = [pair.sentence.text for pair in pairs]
        image_explan_batch = [pair.image_explan.text for pair in pairs]
        
        inputs = self.tokenizer(text_batch, padding=True, return_tensors='pt').to(self.device)
        image_explan_inputs = self.tokenizer(image_explan_batch, padding=True, return_tensors='pt').to(self.device)
        
        outputs = self._bert_forward_with_image(inputs, image_explan_inputs, pairs)
        feats = outputs.last_hidden_state[:, CLS_POS]
        logits = self.aux_head(feats)

        labels = torch.tensor([pair.label for pair in pairs], dtype=torch.long, device=self.device)
        loss = F.cross_entropy(logits, labels, reduction='mean')
        pred = torch.argmax(logits, dim=1).tolist()

        return loss, pred

