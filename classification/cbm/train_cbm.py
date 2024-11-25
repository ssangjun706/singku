import torch
import os
import random
import utils
import data_utils
import similarity
import argparse
import datetime
import json

from tqdm import tqdm
from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from torch.utils.data import DataLoader, TensorDataset

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="ham10000", choices=["ham10000"])
parser.add_argument("--concept_set", type=str, default=None)
parser.add_argument("--backbone", type=str, default="swinv2-large", choices=["swinv2-large"])
parser.add_argument("--clip_name", type=str, default="ViT-B/16", choices=["ViT-B/16"])

parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size used when saving model/CLIP activations")
parser.add_argument("--saga_batch_size", type=int, default=64, help="Batch size used when fitting final layer")
parser.add_argument("--proj_batch_size", type=int, default=64, help="Batch size to use when learning projection layer")

parser.add_argument("--feature_layer", type=str, default='swinv2')
parser.add_argument("--activation_dir", type=str, default='saved_activations')
parser.add_argument("--save_dir", type=str, default='saved_models')
parser.add_argument("--clip_cutoff", type=float, default=0.25, help="concepts with smaller top5 clip activation will be deleted")
parser.add_argument("--proj_steps", type=int, default=1000, help="how many steps to train the projection layer for")
parser.add_argument("--interpretability_cutoff", type=float, default=0.45, help="concepts with smaller similarity to target concept will be deleted")
parser.add_argument("--lam", type=float, default=0.0007, help="Sparsity regularization parameter, higher->more sparse")
parser.add_argument("--n_iters", type=int, default=1000, help="How many iterations to run the final layer solver for")
parser.add_argument("--print", action='store_true', help="Print all concepts being deleted in this stage")

def train_cbm_and_save(args):
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if args.concept_set is None:
        args.concept_set = "./data/concept_sets/{}_filtered.json".format(args.dataset)
        
    similarity_fn = similarity.cos_similarity_cubed_single
    
    d_train = "/DATA/jincheol/dataset/HAM10000_split/train" #데이터셋 경로
    d_val = "/DATA/jincheol/dataset/HAM10000_split/val"
    
    cls_file = data_utils.LABEL_FILES[args.dataset]
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")
    
    with open(args.concept_set) as f:
        concepts = json.load(f)
    
    #save activations and get save_paths
    for d_probe in [d_train, d_val]:
        utils.save_activations(clip_name = args.clip_name, target_name = args.backbone, 
                               target_layers = [args.feature_layer], d_probe = d_probe,
                               concept_set = args.concept_set, batch_size = args.batch_size, 
                               device = args.device, pool_mode = "avg", save_dir = args.activation_dir)
        
    target_save_name, clip_save_name, text_save_name = utils.get_save_names(args.clip_name, args.backbone, 
                                            args.feature_layer,d_train, args.concept_set, "avg", args.activation_dir)
    val_target_save_name, val_clip_save_name, text_save_name =  utils.get_save_names(args.clip_name, args.backbone,
                                            args.feature_layer, d_val, args.concept_set, "avg", args.activation_dir)
    
    #load features
    with torch.no_grad():
        target_features = torch.load(target_save_name, map_location="cpu", weights_only=False).float()
        val_target_features = torch.load(val_target_save_name, map_location="cpu", weights_only=False).float()
    
        image_features = torch.load(clip_save_name, map_location="cpu", weights_only=False).float()
        image_features /= torch.norm(image_features, dim=1, keepdim=True)

        val_image_features = torch.load(val_clip_save_name, map_location="cpu", weights_only=False).float()
        val_image_features /= torch.norm(val_image_features, dim=1, keepdim=True)

        text_features = torch.load(text_save_name, map_location="cpu", weights_only=False).float()
        text_features /= torch.norm(text_features, dim=1, keepdim=True)

        clip_features = image_features @ text_features.T
        val_clip_features = val_image_features @ text_features.T
        del image_features, text_features, val_image_features
    
    # filter concepts not activating highly
    highest = torch.mean(torch.topk(clip_features, dim=0, k=5)[0], dim=0)

    if args.print:
        for i, concept in enumerate(concepts):
            if highest[i] <= args.clip_cutoff:
                print("Deleting {}, CLIP top5:{:.3f}".format(concept, highest[i]))
    concepts = [concepts[i] for i in range(len(concepts)) if highest[i] > args.clip_cutoff]
    del clip_features

    with torch.no_grad():
        image_features = torch.load(clip_save_name, map_location="cpu", weights_only=False).float()
        image_features /= torch.norm(image_features, dim=1, keepdim=True)
        text_features = torch.load(text_save_name, map_location="cpu", weights_only=False).float()[highest > args.clip_cutoff]
        text_features /= torch.norm(text_features, dim=1, keepdim=True)
        clip_features = image_features @ text_features.T
        del image_features, text_features
    
    val_clip_features = val_clip_features[:, highest>args.clip_cutoff]
    
    #learn projection layer
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(concepts),
                                 bias=False).to(args.device)
    opt = torch.optim.Adam(proj_layer.parameters(), lr=1e-3)
    
    indices = [ind for ind in range(len(target_features))]
    
    

    best_val_loss = float("inf")
    best_step = 0
    best_weights = None
    proj_batch_size = min(args.proj_batch_size, len(target_features))
    for i in tqdm(range(args.proj_steps), desc="proj"):
        batch = torch.LongTensor(random.sample(indices, k=proj_batch_size))
        outs = proj_layer(target_features[batch].to(args.device).detach())
        loss = -similarity_fn(clip_features[batch].to(args.device).detach(), outs)
        
        loss = torch.mean(loss)
        loss.backward()
        opt.step()
        if i%50==0 or i==args.proj_steps-1:
            with torch.no_grad():
                val_output = proj_layer(val_target_features.to(args.device).detach())
                val_loss = -similarity_fn(val_clip_features.to(args.device).detach(), val_output)
                val_loss = torch.mean(val_loss)
            if i==0:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
                print("Step:{}, Avg train similarity:{:.4f}, Avg val similarity:{:.4f}".format(best_step, -loss.cpu(),
                                                                                               -best_val_loss.cpu()))
                
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
            else: #stop if val loss starts increasing
                break
        opt.zero_grad()
        
    proj_layer.load_state_dict({"weight":best_weights})
    print("Best step:{}, Avg val similarity:{:.4f}".format(best_step, -best_val_loss.cpu()))
    
    #delete concepts that are not interpretable
    with torch.no_grad():
        outs = proj_layer(val_target_features.to(args.device).detach())
        sim = similarity_fn(val_clip_features.to(args.device).detach(), outs)
        interpretable = sim > args.interpretability_cutoff
        
    if args.print:
        for i, concept in enumerate(concepts):
            if sim[i]<=args.interpretability_cutoff:
                print("Deleting {}, Iterpretability:{:.3f}".format(concept, sim[i]))
    
    concepts = [concepts[i] for i in range(len(concepts)) if interpretable[i]]
    
    del clip_features, val_clip_features
    
    W_c = proj_layer.weight[interpretable]
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(concepts), bias=False)
    proj_layer.load_state_dict({"weight":W_c})
    
    train_targets = data_utils.get_targets_only(d_train)
    val_targets = data_utils.get_targets_only(d_val)
        
    
    with torch.no_grad():
        train_c = proj_layer(target_features.detach())
        val_c = proj_layer(val_target_features.detach())
        
        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)
        
        train_c -= train_mean
        train_c /= train_std
        
        train_y = torch.LongTensor(train_targets)
        indexed_train_ds = IndexedTensorDataset(train_c, train_y)

        val_c -= train_mean
        val_c /= train_std
        
        val_y = torch.LongTensor(val_targets)

        val_ds = TensorDataset(val_c,val_y)

    
    _, train_counts = torch.unique(train_y, return_counts=True)
    _, val_counts = torch.unique(val_y, return_counts=True)

    #class_불균형
    #class 가중치 추가
    
    # train_counts: tensor([ 834, 5028,  385,  245,  824,   86,  106])
    # val_counts: tensor([112, 671,  51,  32, 110,  11,  14])
    
    
    # 클래스 가중치 계산
    class_weights_train = 1.0 / train_counts
    class_weights_val = 1.0 / val_counts
    
    # 정규화 (선택 사항)
    class_weights = class_weights_train / class_weights_train.sum()
    class_weights=class_weights.to(args.device)
    #class_weights_val = class_weights_val / class_weights_val.sum()
    
    # class_weights_train: tensor([0.0395, 0.0066, 0.0856, 0.1345, 0.0400, 0.3831, 0.3108])
    # class_weights_val: tensor([0.0384, 0.0064, 0.0843, 0.1343, 0.0391, 0.3907, 0.3069])
    
    #print(f'train_counts: {class_weights.shape}')
    # print(f'val_counts: {class_weights_val}')
    
    
    #breakpoint()
    
    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)

   

    # Make linear model and zero initialize
    linear = torch.nn.Linear(train_c.shape[1],len(classes)).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    
    STEP_SIZE = 0.1
    ALPHA = 0.99
    metadata = {}
    metadata['max_reg'] = {}
    metadata['max_reg']['nongrouped'] = args.lam

    # Solve the GLM path
    output_proj = glm_saga(linear, indexed_train_loader, STEP_SIZE, args.n_iters, ALPHA,class_weights, epsilon=1, k=1,
                      val_loader=val_loader, do_zero=False, metadata=metadata, n_ex=len(target_features), n_classes = len(classes))
    W_g = output_proj['path'][0]['weight']
    b_g = output_proj['path'][0]['bias']
    
    save_name = "{}/{}_cbm_{}".format(args.save_dir, args.dataset, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
    os.mkdir(save_name)
    torch.save(train_mean, os.path.join(save_name, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_name, "proj_std.pt"))
    torch.save(W_c, os.path.join(save_name ,"W_c.pt"))
    torch.save(W_g, os.path.join(save_name, "W_g.pt"))
    torch.save(b_g, os.path.join(save_name, "b_g.pt"))
    
    with open(os.path.join(save_name, "concepts.txt"), 'w') as f:
        f.write(concepts[0])
        for concept in concepts[1:]:
            f.write('\n'+concept)
    
    with open(os.path.join(save_name, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    with open(os.path.join(save_name, "metrics.txt"), 'w') as f:
        out_dict = {}
        for key in ('lam', 'lr', 'alpha', 'time'):
            out_dict[key] = float(output_proj['path'][0][key])
        out_dict['metrics'] = output_proj['path'][0]['metrics']
        nnz = (W_g.abs() > 1e-5).sum().item()
        total = W_g.numel()
        out_dict['sparsity'] = {"Non-zero weights":nnz, "Total weights":total, "Percentage non-zero":nnz/total}
        json.dump(out_dict, f, indent=2)
    
if __name__=='__main__':
    args = parser.parse_args()
    train_cbm_and_save(args)