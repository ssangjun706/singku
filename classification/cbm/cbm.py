import os
import json
import torch
import data_utils

class CBM_model(torch.nn.Module):
    def __init__(self, backbone_name, W_c, W_g, b_g, proj_mean, proj_std, device="cuda"):
        super().__init__()
        model, _ = data_utils.get_target_model(backbone_name, device)
        #remove final fully connected layer
        if "clip" in backbone_name:
            self.backbone = model
        elif "cub" in backbone_name:
            self.backbone = lambda x: model.features(x)
        else:
            self.backbone = torch.nn.Sequential(*list(model.children())[:-1])
            #self.backbone = torch.nn.Sequential(*list(model.children()))
            
        self.proj_layer = torch.nn.Linear(in_features=W_c.shape[1], out_features=W_c.shape[0], bias=False).to(device)
        self.proj_layer.load_state_dict({"weight":W_c})
            
        self.proj_mean = proj_mean
        self.proj_std = proj_std
        
        self.final = torch.nn.Linear(in_features = W_g.shape[1], out_features=W_g.shape[0]).to(device)
        self.final.load_state_dict({"weight":W_g, "bias":b_g})
        self.concepts = None
        
    def forward(self, x):
        #print(f'x_shape: {x.shape}')
      
        #x = self.backbone[0](x)  # Swinv2Model 실행
    
        # Swinv2ModelOutput에서 pooler_output 추출
        # if hasattr(x, "pooler_output"):
        #     x = x.pooler_output  # Global Average Pooling 결과 사용
        # elif hasattr(x, "last_hidden_state"):
        #     x = torch.mean(x.last_hidden_state, dim=1)  # 평균 풀링
        # else:
        #     raise ValueError("Unexpected Swinv2ModelOutput structure")
    
      
    
        # classifier 실행
        #x = self.backbone[1](x)  # Linear 레이어
        #print(f"Classifier output shape: {x.shape}")
        #x = x.pooler_output
        #[25*1536] proj_layer랑 내적하려할때, [1000*171]임
        
        
        
        x=self.backbone(x)
        x = x.pooler_output 
        x = torch.flatten(x, 1)
        x = self.proj_layer(x)
        proj_c = (x-self.proj_mean)/self.proj_std
        x = self.final(proj_c)
        return x, proj_c

    
def load_cbm(load_dir, device):
    with open(os.path.join(load_dir ,"args.txt"), 'r') as f:
        args = json.load(f)

    W_c = torch.load(os.path.join(load_dir ,"W_c.pt"), map_location=device)
    W_g = torch.load(os.path.join(load_dir, "W_g.pt"), map_location=device)
    b_g = torch.load(os.path.join(load_dir, "b_g.pt"), map_location=device)

    proj_mean = torch.load(os.path.join(load_dir, "proj_mean.pt"), map_location=device)
    proj_std = torch.load(os.path.join(load_dir, "proj_std.pt"), map_location=device)

    model = CBM_model(args['backbone'], W_c, W_g, b_g, proj_mean, proj_std, device)
    return model