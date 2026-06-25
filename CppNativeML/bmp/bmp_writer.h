//
//  bmp_writer.h
//  CSE302
//
//  Created by Sora Sugiyama on 3/18/25.
//  Modified the code of https://stackoverflow.com/questions/2654480/writing-bmp-image-in-pure-c-c-without-other-libraries Answer by Morgoth and deusmacbre
//

#ifndef bmp_writer_h
#define bmp_writer_h

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <vector>
#include <cassert>
#include <string>

namespace BMP{

using u8=u_int8_t;

class lum{
    std::vector<std::vector<u8> > _vuc;
public:
    size_t n,m;
    lum(size_t N=0,size_t M=0){
        n=N;m=M;
        _vuc=std::vector<std::vector<u8> >(n,std::vector<u8>(m));
    }
    ~lum(){
        n=m=0;
    }
    
    u8& operator()(size_t i,size_t j){
        return _vuc[i][j];
    }
};

static u8 bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
static u8 bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
static u8 bmppad[3] = {0,0,0};

class bmp{
public:
    lum red,green,blue;
    
    bmp(size_t N=0,size_t M=0){
        red=green=blue=lum(N,M);
    }
    ~bmp(){}
    
    void init(lum &R,lum &G,lum &B){
        assert(("All size of matrices must be same.\n"&&
                R.n==G.n&&G.n==B.n&&
                R.m==G.m&&G.m==B.m));
        red=R;
        green=G;
        blue=B;
    }
    
    void writeBMP(std::string path,std::string filename){
        FILE *f;
        u8 *img=NULL;
        size_t w=red.n,h=red.m;
        size_t filesize=54 + 3*h*w;
        
        img=(u8*)malloc(3*h*w);
        memset(img,0,3*h*w);
        
        for(size_t i=0;i<w;i++){
            for(size_t j=0;j<h;j++){
                size_t x=i,y=h-j-1;
                img[(x+y*w)*3+2]=red(i,j);
                img[(x+y*w)*3+1]=green(i,j);
                img[(x+y*w)*3+0]=blue(i,j);
            }
        }
        
        bmpfileheader[ 2] = (u8)(filesize    );
        bmpfileheader[ 3] = (u8)(filesize>> 8);
        bmpfileheader[ 4] = (u8)(filesize>>16);
        bmpfileheader[ 5] = (u8)(filesize>>24);
        
        bmpinfoheader[ 4] = (u8)(       w    );
        bmpinfoheader[ 5] = (u8)(       w>> 8);
        bmpinfoheader[ 6] = (u8)(       w>>16);
        bmpinfoheader[ 7] = (u8)(       w>>24);
        bmpinfoheader[ 8] = (u8)(       h    );
        bmpinfoheader[ 9] = (u8)(       h>> 8);
        bmpinfoheader[10] = (u8)(       h>>16);
        bmpinfoheader[11] = (u8)(       h>>24);
        
        if(path.back()!='/')path.push_back('/');
        f = fopen((path+filename).c_str(),"wb");
        fwrite(bmpfileheader,1,14,f);
        fwrite(bmpinfoheader,1,40,f);
        for(int i=0; i<h; i++)
        {
            fwrite(img+(w*(h-i-1)*3),3,w,f);
            fwrite(bmppad,1,(4-(w*3)%4)%4,f);
        }
        
        free(img);
        fclose(f);
    }
};
}


#endif /* bmp_writer_h */
