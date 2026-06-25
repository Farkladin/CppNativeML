//
//  bmplot.h
//  CSE302
//
//  Created by Sora Sugiyama on 3/18/25.
//

#ifndef bmplot_h
#define bmplot_h

#include "bmp/bmp_writer.h"
#include "bmp/writeNumber.h"
#include <algorithm>
#include <cmath>
#include <cassert>

namespace BMP{

class fnct_plotter{
    int ox,oy,w,h;
    double Mx,My,mx,my;
    int scale_x,scale_y;
    bmp paper;
    
public:
    fnct_plotter(double min_x=-1,double max_x=1,double min_y=-1,double max_y=1,int W=1024,int H=600){
        
        assert(("W and H are to small\n"&&W>=300&&H>=300));
        
        scale_x=int((W-100)/std::max((max_x-min_x),1.0));
        scale_y=int((H-100)/std::max((max_y-min_y),1.0));
        int x0=50-int(min_x*scale_x),y0=50-int(min_y*scale_y);
        
        ox=x0;oy=y0;Mx=max_x;My=max_y;mx=min_x;my=min_y;
        w=W;h=H;
        paper=bmp(W,H);
        
        lum &R=paper.red,&B=paper.blue,&G=paper.green;
        for(int i=50;i<W-50;i++){
            R(i,50)=G(i,50)=B(i,50)=255;
            R(i,H-50)=G(i,H-50)=B(i,H-50)=255;
        }
        for(int j=50;j<H-50;j++){
            R(50,j)=G(50,j)=B(50,j)=255;
            R(W-50,j)=G(W-50,j)=B(W-50,j)=255;
        }
        
        if(50<=y0&&y0<=H-50){
            writeDouble(paper,0,10,y0-10,2);
            for(int i=50;i<54;i++)R(i,y0)=G(i,y0)=B(i,y0)=255;
        }
        if(50<=x0&&x0<=W-50){
            writeDouble(paper,0,x0-10,30,2);
            for(int i=50;i<54;i++)R(x0,i)=G(x0,i)=B(x0,i)=255;
        }
        
        int X=W-50,Y=H-50;
        if(y0!=H-50)writeDouble(paper,max_y,10,Y,2);
        if(y0!=50)writeDouble(paper,min_y,10,50,2);
        if(x0!=W-50)writeDouble(paper,max_x,X-5,30,2);
        if(x0!=50)writeDouble(paper,min_x,45,30,2);
    }
    ~fnct_plotter(){}
    
public:
    
    // if domain of function is (a,b), then lo=a, hi=b;
    template<class F>
    void plotFnct(F fnct,double lo,double hi,int thick=2,u8 r=255,u8 g=255,u8 b=255){
        
        lum &R=paper.red,&B=paper.blue,&G=paper.green;
        
        lo=std::max(lo,mx);
        hi=std::min(hi,Mx);
        int le=int(lo*scale_x),ri=int(hi*scale_x);
        for(int i=le;i<=ri;i++){
            double curd=(fnct(double(i)/scale_x))*scale_y;
            int cur=curd+(curd<0?-0.5:0.5)+oy;
            for(int j=0;j<thick;j++){
                if(cur+j>50&&j+cur<h-50){
                    R(ox+i,j+cur)=r;
                    G(ox+i,j+cur)=g;
                    B(ox+i,j+cur)=b;
                }
                if(cur-j>50&&cur-j<h-50){
                    R(ox+i,cur-j)=r;
                    G(ox+i,cur-j)=g;
                    B(ox+i,cur-j)=b;
                }
            }
        }
    }
    
    void plotScatter(std::vector<std::pair<double,double> >&points,int radi,u8 r=255,u8 g=255,u8 b=255){
        lum &R=paper.red,&B=paper.blue,&G=paper.green;
        
        for(auto &p:points){
            int x=p.first*scale_x+(p.first<0?-0.5:0.5)+ox,y=p.second*scale_y+(p.second<0?-0.5:0.5)+oy;
            for(int i=x-radi;i<=x+radi;i++){
                for(int j=y-radi;j<=y+radi;j++){
                    if((x-i)*(x-i)+(y-j)*(y-j)>radi)continue;
                    if(50<=i&&i<=w-50&&50<=j&&j<=h-50){
                        R(i,j)=r;
                        G(i,j)=g;
                        B(i,j)=b;
                    }
                }
            }
        }
    }
    
    void plot(std::string path,std::string filename){
        paper.writeBMP(path,filename);
    }
};
}
#endif /* bmplot_h */
