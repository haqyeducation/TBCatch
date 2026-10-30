"""Patient-weighted, masked attention MIL; GPU training, portable CPU inference."""
import numpy as np
import torch
from torch import nn


class BagNetwork(nn.Module):
    def __init__(self,audio_dim,tab_dim,hidden,dropout):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(audio_dim,hidden),nn.ReLU(),nn.Dropout(dropout))
        self.attention=nn.Sequential(nn.Linear(hidden+tab_dim,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.head=nn.Sequential(nn.Linear(hidden+tab_dim,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,1))

    def forward(self,a,mask,t):
        h=self.encoder(a);context=t[:,None,:].expand(-1,a.shape[1],-1)
        score=self.attention(torch.cat([h,context],dim=-1)).squeeze(-1).masked_fill(~mask,-1e9)
        weight=torch.softmax(score,dim=1)
        pooled=(h*weight[:,:,None]).sum(dim=1)
        return self.head(torch.cat([pooled,t],dim=1)).squeeze(-1),weight


def batch(bags,tab,indices,device):
    chosen=[bags[i] for i in indices];length=max(map(len,chosen))
    a=np.zeros((len(chosen),length,chosen[0].shape[1]),dtype=np.float32);mask=np.zeros(a.shape[:2],bool)
    for j,b in enumerate(chosen):a[j,:len(b)]=b;mask[j,:len(b)]=True
    return (torch.as_tensor(a,device=device),torch.as_tensor(mask,device=device),
            torch.as_tensor(np.asarray(tab[indices],dtype=np.float32),device=device))


def fit_attention(bags,tab,y,candidate,cfg,seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    device='cuda' if cfg.get('attention_device','auto')!='cpu' and torch.cuda.is_available() else 'cpu'
    dims=dict(audio_dim=bags[0].shape[1],tab_dim=tab.shape[1],hidden=candidate['hidden'],dropout=candidate['dropout'])
    net=BagNetwork(**dims).to(device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=candidate['learning_rate'],weight_decay=candidate['weight_decay'])
    # Fixed epochs selected in inner CV; no outer calibration/test early stopping.
    lossfn=nn.BCEWithLogitsLoss();rng=np.random.default_rng(seed);history=[]
    for epoch in range(candidate['epochs']):
        net.train();order=rng.permutation(len(y));total=0.
        for start in range(0,len(y),cfg['attention_batch_size']):
            ix=order[start:start+cfg['attention_batch_size']];a,mask,t=batch(bags,tab,ix,device)
            target=torch.as_tensor(y[ix],dtype=torch.float32,device=device)
            optimizer.zero_grad(set_to_none=True);logits,_=net(a,mask,t);loss=lossfn(logits,target)
            loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step()
            total+=float(loss.detach())*len(ix)
        history.append(total/len(y))
    state={k:v.detach().cpu().numpy().copy() for k,v in net.state_dict().items()}
    return dict(dims=dims,state=state,epochs=candidate['epochs'],training_loss=history,training_device=device)


def predict_attention(saved,bags,tab,weights=False):
    # Float64 CPU replay makes Shapley completeness stable across batch sizes
    # and differing padding lengths; saved training weights remain float32.
    net=BagNetwork(**saved['dims']);net.load_state_dict({k:torch.as_tensor(v) for k,v in saved['state'].items()})
    net=net.double();net.eval()
    probabilities=[];allweights=[]
    with torch.no_grad():
        for start in range(0,len(bags),128):
            ix=np.arange(start,min(start+128,len(bags)));a,mask,t=batch(bags,tab,ix,'cpu')
            logits,w=net(a.double(),mask,t.double());probabilities.extend(torch.sigmoid(logits).numpy())
            allweights.extend(w[j,:len(bags[i])].numpy() for j,i in enumerate(ix))
    return allweights if weights else np.asarray(probabilities,dtype=float)
