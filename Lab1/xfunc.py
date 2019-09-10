def xfunc(t,xvec):
    if t==0:
        return 1.5
    else:
        x1=xvec[t]

        if (t-26) <0:
            x25 = 0
        else:
            x25=xvec[t-25]

        return x1+0.2*x25/(1+x25**10)-0.1*x1
