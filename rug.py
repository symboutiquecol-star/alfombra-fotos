import numpy as np, cv2, sys, time
W,H=int(sys.argv[1]),int(sys.argv[2]); out=sys.argv[3]; seed=int(sys.argv[4]) if len(sys.argv)>4 else 7
rng=np.random.default_rng(seed)
sc=W/3024
def noise(cell):
    g=rng.normal(0,1,(int(H/cell)+3,int(W/cell)+3)).astype(np.float32)
    return cv2.resize(cv2.GaussianBlur(g,(0,0),0.8),(W,H),interpolation=cv2.INTER_CUBIC)
hgt=noise(210*sc)+0.5*noise(95*sc)
hgt=(hgt-hgt.mean())/hgt.std()
gy,gx=np.gradient(cv2.GaussianBlur(hgt,(0,0),6*sc)*60*sc)
light=np.clip(1+0.50*(-(gx+gy)/1.414),0.6,1.4)
ao=np.clip(0.82+0.15*hgt,0.55,1.12)
macro=(light*ao).astype(np.float32)
shade=np.full((H,W),0.40,np.float32)
t=time.time()
ntufts=int(13000*(W*H)/(3024*4032))
pts=np.column_stack([rng.uniform(-200,W+200,ntufts),rng.uniform(-200,H+200,ntufts)])
pts=pts[np.argsort(pts[:,1])]
nseg=10
off=np.array([5*sc,7*sc])
for (x0,y0) in pts:
    base_ang=rng.uniform(0,2*np.pi); tb=rng.uniform(0.9,1.05)
    fibers=[]
    for k in range(rng.integers(12,20)):
        ang=base_ang+rng.normal(0,0.5)
        L=rng.uniform(140,260)*sc
        curv=rng.normal(0,0.010)/sc
        th0=rng.uniform(8,12)*sc
        x,y=x0+rng.normal(0,14*sc),y0+rng.normal(0,14*sc)
        P=[(x,y)]; a=ang
        for i in range(nseg):
            a+=curv*L/nseg+rng.normal(0,0.04); x+=np.cos(a)*L/nseg; y+=np.sin(a)*L/nseg; P.append((x,y))
        fibers.append((np.array(P),th0,ang))
    for P,th0,ang in fibers:
        cv2.polylines(shade,[np.round(P+off).astype(np.int32)],False,float(0.30*tb),int(th0*1.3)+1,cv2.LINE_AA)
    for P,th0,ang in fibers:
        Pi=np.round(P).astype(np.int32)
        lit=0.92+0.12*(-(np.cos(ang)+np.sin(ang))/1.414)
        tipb=tb*lit*rng.uniform(0.92,1.0)
        for i in range(nseg):
            f=(i+1)/nseg
            v=0.45*tb+(tipb-0.45*tb)*f**0.7
            th=max(1,int(round(th0*(1-0.35*f**3))))
            cv2.line(shade,tuple(Pi[i]),tuple(Pi[i+1]),float(v),th,cv2.LINE_AA)
        Q=np.round(P[5:]+np.array([-0.25*th0,-0.25*th0])).astype(np.int32)
        cv2.polylines(shade,[Q],False,float(min(1.2,tipb+0.16)),max(1,int(th0*0.3)),cv2.LINE_AA)
print('strands',round(time.time()-t,1))
shade=cv2.GaussianBlur(shade,(0,0),max(0.6,2.0*sc))*macro
np.save(out,shade.astype(np.float16))
