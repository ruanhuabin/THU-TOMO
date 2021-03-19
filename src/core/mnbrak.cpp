#include <cmath>
#include "nr.h"
using namespace std;

namespace {
	inline void shft3(DP &a, DP &b, DP &c, const DP d)
	{
		a=b;
		b=c;
		c=d;
	}
}

void NR::mnbrak(DP &ax, DP &bx, DP &cx, DP &fa, DP &fb, DP &fc,
	DP func(const DP,vector<Track> &,float *,float *,float,int,float*,float*),vector<Track> &tracks,float *d_x,float *d_y,float psi,int ref,float *c_x,float *c_y)
{
	const DP GOLD=1.618034,GLIMIT=100.0,TINY=1.0e-20;
	DP ulim,u,r,q,fu;

/*	fa=func(ax,tracks,d_x,d_y,psi,ref);
	fb=func(bx,tracks,d_x,d_y,psi,ref);
	if (fb > fa) {
		SWAP(ax,bx);
		SWAP(fb,fa);
	}
	cx=bx+GOLD*(bx-ax);
	fc=func(cx,tracks,d_x,d_y,psi,ref);
	while (fb > fc) {
		r=(bx-ax)*(fb-fc);
		q=(bx-cx)*(fb-fa);
		u=bx-((bx-cx)*q-(bx-ax)*r)/
			(2.0*SIGN(MAX(fabs(q-r),TINY),q-r));
		ulim=bx+GLIMIT*(cx-bx);
		if ((bx-u)*(u-cx) > 0.0) {
			fu=func(u,tracks,d_x,d_y,psi,ref);
			if (fu < fc) {
				ax=bx;
				bx=u;
				fa=fb;
				fb=fu;
				return;
			} else if (fu > fb) {
				cx=u;
				fc=fu;
				return;
			}
			u=cx+GOLD*(cx-bx);
			fu=func(u,tracks,d_x,d_y,psi,ref);
		} else if ((cx-u)*(u-ulim) > 0.0) {
			fu=func(u,tracks,d_x,d_y,psi,ref);
			if (fu < fc) {
				shft3(bx,cx,u,cx+GOLD*(cx-bx));
				shft3(fb,fc,fu,func(u,tracks,d_x,d_y,psi,ref));
			}
		} else if ((u-ulim)*(ulim-cx) >= 0.0) {
			u=ulim;
			fu=func(u,tracks,d_x,d_y,psi,ref);
		} else {
			u=cx+GOLD*(cx-bx);
			fu=func(u,tracks,d_x,d_y,psi,ref);
		}
		shft3(ax,bx,cx,u);
		shft3(fa,fb,fc,fu);
	}*/

	fa=func(ax,tracks,d_x,d_y,psi,ref,c_x,c_y);
	fb=func(bx,tracks,d_x,d_y,psi,ref,c_x,c_y);
	if (fb > fa) {
		SWAP(ax,bx);
		SWAP(fb,fa);
	}
	cx=bx+GOLD*(bx-ax);
	fc=func(cx,tracks,d_x,d_y,psi,ref,c_x,c_y);
	while(fc<fb)
	{
		ax=bx;
		bx=cx;
		cx=bx+GOLD*(bx-ax);
		fa=func(ax,tracks,d_x,d_y,psi,ref,c_x,c_y);
		fb=func(bx,tracks,d_x,d_y,psi,ref,c_x,c_y);
		fc=func(cx,tracks,d_x,d_y,psi,ref,c_x,c_y);
	}
}
