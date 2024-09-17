import torch
import models.models_helper as models_helper
import models.MambaST as MambaST
import torch.nn as nn
import models.mamba as Mamba

def load_pretrained(args):
    vgg = models_helper.vgg
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:44])

    decoder = models_helper.decoder
    mamba = Mamba.Mamba(args=args)
    decoder_path = args.decoder_path
    mamba_path = args.mamba_path
    embedding_path = args.embedding_path
        
    embedding = models_helper.PatchEmbed()

    decoder.eval()
    mamba.eval()
    vgg.eval()
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    state_dict = torch.load(decoder_path)
    for k, v in state_dict.items():
        namekey = k
        new_state_dict[namekey] = v
    decoder.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    state_dict = torch.load(mamba_path)
    for k, v in state_dict.items():
        namekey = k
        new_state_dict[namekey] = v
    mamba.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    state_dict = torch.load(embedding_path)
    for k, v in state_dict.items():
        namekey = k
        new_state_dict[namekey] = v
    embedding.load_state_dict(new_state_dict)

    with torch.no_grad():
        network = MambaST.MambaST(vgg,decoder,embedding,mamba,args)
    
    print(f"Loaded Embedding checkpoints from {embedding_path}")
    print(f"Loaded Mamba checkpoints from {mamba_path}")
    print(f"Loaded CNN decoder checkpoints from {decoder_path}")
    return network