"""Input-response penalty; all atmospheric models remain frozen."""
import copy,torch

def response_loss(model,x,base,available,historical_valid,sd,use_cot=True):
    assert x.requires_grad
    z=x
    if not use_cot:
        z=x.clone();z[:,279:297]=0
    prediction=model(z,base,available)
    grad=torch.autograd.grad(prediction.sum(),x,create_graph=True)[0]
    direction=(grad[:,279:288]+grad[:,288:297]*historical_valid).sum(1)/sd
    penalty=torch.relu(direction).square().mean()
    return prediction,penalty,direction

def engineering_checks(template,features,base,available,historical_valid,sd):
    device=features.device
    model=copy.deepcopy(template).double()
    with torch.no_grad():
        for p in model.parameters():p.zero_()
        model.delta[0].weight[0,279:288]=1
        model.delta[-1].weight[0,0]=1
    x=torch.zeros((32,features.shape[1]),device=device,dtype=torch.float64)
    x[:,279:288]=.2;x=x.requires_grad_(True)
    b=torch.full((32,),2.,device=device,dtype=torch.float64);av=torch.ones_like(b);hv=torch.ones((32,9),device=device,dtype=torch.float64)
    y,omega,d=response_loss(model,x,b,av,hv,sd)
    assert omega.item()>0
    grads=torch.autograd.grad(omega,tuple(model.parameters()),allow_unused=True)
    assert all(g is None or torch.isfinite(g).all() for g in grads)
    norm=sum(float(g.abs().sum()) for g in grads if g is not None);assert norm>0
    v=torch.zeros_like(x);v[:,279:297]=1/sd
    with torch.no_grad():fd=(model(x+1e-5*v,b,av)-model(x-1e-5*v,b,av))/(2e-5)
    error=float(abs(fd-d).max());assert error<1e-5
    _,neutral,_=response_loss(model,x,b,av,hv,sd,False)
    zero_grads=torch.autograd.grad(neutral,tuple(model.parameters()),allow_unused=True)
    assert neutral.item()==0 and all(g is None or torch.count_nonzero(g)==0 for g in zero_grads)
    missing=torch.zeros_like(av);yy,oo,dd=response_loss(model,x,b,missing,hv,sd)
    assert torch.equal(yy,b) and oo.item()==0 and torch.count_nonzero(dd)==0
    # Actual small batch, finite graph, no parameter updates.
    real=copy.deepcopy(template)
    xx=features[:32].detach().requires_grad_(True)
    yy,oo,dd=response_loss(real,xx,base[:32],available[:32],historical_valid[:32],sd)
    assert torch.isfinite(oo) and torch.isfinite(dd).all()
    assert torch.equal(yy,base[:32]),'zero-initialized head must exactly reproduce G1'
    # One complete optimizer/scheduler checkpoint, then identical second update.
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60,eta_min=1e-5)
    def update(m,o,s):
        xx=x.detach().requires_grad_(True);y,reg,_=response_loss(m,xx,b,av,hv,sd)
        loss=(y-1).square().mean()+reg;o.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);o.step();s.step()
        return float(loss.detach())
    update(model,opt,scheduler)
    state=copy.deepcopy(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),step=1))
    restored=copy.deepcopy(model);ro=torch.optim.AdamW(restored.parameters(),lr=1e-3,weight_decay=1e-4);rs=torch.optim.lr_scheduler.CosineAnnealingLR(ro,T_max=60,eta_min=1e-5)
    restored.load_state_dict(state['model']);ro.load_state_dict(state['optimizer']);rs.load_state_dict(state['scheduler'])
    loss1=update(model,opt,scheduler);loss2=update(restored,ro,rs)
    assert loss1==loss2 and all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
    assert scheduler.state_dict()==rs.state_dict() and all(int(s['step'])==2 for s in ro.state.values())
    return dict(state='PASS',finite_difference_max=error,second_order_gradient_L1=norm,
        no_COT_penalty=0,missing_penalty=0,resume_loss_difference=abs(loss1-loss2),restored_updates=2)
