"""Source-only NJ + ER Mk imputation. No external phylogeny is asserted."""
import numpy as np
from scipy.optimize import minimize_scalar

def known(a):
    a=np.asarray(a)
    return np.isin(a,list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'))

def distance_totals(a):
    k=known(a)
    valid=k[:,None,:]&k[None,:,:]
    return valid.sum(2), ((a[:,None,:]!=a[None,:,:])&valid).sum(2)

def excluding_distance(a,j,totals):
    nobs,mismatch=totals
    kj=known(a[:,j]); both=kj[:,None]&kj[None,:]
    den=nobs-both
    num=mismatch-((a[:,j,None]!=a[None,:,j])&both)
    d=np.divide(num,den,out=np.full(num.shape,np.nan),where=den>0)
    upper=np.triu(np.ones(d.shape,dtype=bool),1)
    valid=upper&np.isfinite(d)
    fallback=float(d[valid].mean()) if valid.any() else .5
    zero=int((upper&~np.isfinite(d)).sum())
    d[~np.isfinite(d)]=fallback
    np.fill_diagonal(d,0)
    return d,zero

def neighbor_join(d):
    d=d.copy();n=len(d);active=list(range(n));edges=[];next_node=n
    negative=0;truncated=0
    while len(active)>2:
        m=len(active);s=d.sum(1);q=(m-2)*d-s[:,None]-s[None,:]
        q[np.tril_indices(m)]=np.inf
        i,j=np.unravel_index(np.argmin(q),q.shape)
        li=.5*d[i,j]+(s[i]-s[j])/(2*(m-2));lj=d[i,j]-li
        for old,length in [(active[i],li),(active[j],lj)]:
            negative+=int(length<0);truncated+=int(length<1e-8)
            edges.append((next_node,old,max(float(length),1e-8)))
        keep=[v for v in range(m) if v not in (i,j)]
        nd=(d[i,keep]+d[j,keep]-d[i,j])/2
        sub=d[np.ix_(keep,keep)]
        d=np.zeros((m-1,m-1));d[:-1,:-1]=sub;d[-1,:-1]=nd;d[:-1,-1]=nd
        active=[active[v] for v in keep]+[next_node];next_node+=1
    length=float(d[0,1]);negative+=int(length<0);truncated+=int(length<1e-8)
    edges.append((active[0],active[1],max(length,1e-8)))
    return edges, {'negative_branches':negative,'truncated_branches':truncated,'edges':len(edges)}

def orient(edges,root=None):
    count=1+max(max(a,b) for a,b,t in edges)
    adj=[[] for _ in range(count)]
    for a,b,t in edges:adj[a].append((b,t));adj[b].append((a,t))
    root=count-1 if root is None else root
    parent=np.full(count,-2,dtype=int);parent[root]=-1;order=[root]
    lengths=np.zeros(count);children=[[] for _ in range(count)]
    for a in order:
        for b,t in adj[a]:
            if b==parent[a]:continue
            assert parent[b]==-2,'Not a tree'
            parent[b]=a;lengths[b]=t;children[a].append(b);order.append(b)
    assert len(order)==count
    return root,order,children,lengths

def transition(q,t,k):
    e=np.exp(-k*q*t)
    return np.full((k,k),(1-e)/k)+np.eye(k)*e

def pruning(tree,emission,q,return_post=False):
    root,order,children,lengths=tree
    count,k=emission.shape
    up=np.ones_like(emission);msg=np.ones_like(emission)
    logscale=0.
    for a in reversed(order):
        value=emission[a].copy()
        for b in children[a]:value*=msg[b]
        scale=value.max()
        if scale<=0:return (-np.inf,None) if return_post else -np.inf
        up[a]=value/scale;logscale+=np.log(scale)
        if a!=root:msg[a]=transition(q,lengths[a],k)@up[a]
    ll=float(logscale+np.log(up[root].mean()))
    if not return_post:return ll
    outside=np.ones_like(emission);outside[root]=np.repeat(1/k,k)
    post=np.zeros_like(emission)
    for a in order:
        value=outside[a]*up[a];post[a]=value/value.sum()
        for b in children[a]:
            value=outside[a]*emission[a]
            for other in children[a]:
                if other!=b:value=value*msg[other]
            value=value/value.max()
            outside[b]=transition(q,lengths[b],k).T@value
            outside[b]/=outside[b].max()
    return ll,post

def predict_mk(edges,n_tips,labels,train,test):
    classes=np.unique(labels[train]);k=len(classes)
    if k==1:
        return classes,np.ones((len(test),1)),{'constant_fallback':True,'rate':None,'rate_boundary':False,'training_loglik':0.}
    tree=orient(edges);count=len(tree[1]);emission=np.ones((count,k))
    for i in train:emission[i]=(classes==labels[i]).astype(float)
    grid=np.linspace(-8,8,17)
    vals=np.array([pruning(tree,emission,np.exp(x)) for x in grid])
    best=int(np.argmax(vals));lo=grid[max(0,best-1)];hi=grid[min(len(grid)-1,best+1)]
    opt=minimize_scalar(lambda x:-pruning(tree,emission,np.exp(x)),bounds=(lo,hi),method='bounded',options={'xatol':1e-5,'maxiter':80})
    choices=[(vals[best],grid[best]),(-opt.fun,float(opt.x))]
    ll,logq=max(choices)
    _,post=pruning(tree,emission,np.exp(logq),True)
    prob=post[test]
    assert np.isfinite(prob).all() and np.allclose(prob.sum(1),1)
    return classes,prob,{'constant_fallback':False,'rate':float(np.exp(logq)),
                         'rate_boundary':bool(abs(logq)>=7.999),'training_loglik':float(ll)}
