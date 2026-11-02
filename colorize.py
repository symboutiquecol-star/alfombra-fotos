import numpy as np, cv2
def colorize(shade, rgb, floor=0.42):
    c=np.array(rgb,np.float32)/255
    s=shade.astype(np.float32); s=s/np.mean(s)
    s=floor+(1-floor)*s
    s3=s[...,None]
    base=c*np.minimum(s3,1.0)
    hi=np.clip((s3-1.0)/0.3,0,1)
    out=base*(1-hi)+(c*0.6+0.4)*hi
    w=np.clip(1-s3,0,1)
    out=out*((c/c.mean())**0.5)**w
    hsv=cv2.cvtColor(np.clip(out,0,1).astype(np.float32),cv2.COLOR_RGB2HSV)
    hsv[...,1]=np.clip(hsv[...,1]*(1+0.25*np.clip(1-s,0,1)),0,1)
    return np.clip(cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB),0,1)
