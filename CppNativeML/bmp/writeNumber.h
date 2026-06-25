//
//  writeNumber.h
//  CSE302
//
//  Created by Sora Sugiyama on 3/18/25.
//

#ifndef writeNumber_h
#define writeNumber_h

#include <vector>
#include <string>
#include "bmp/bmp_writer.h"

namespace BMP{
const std::vector<std::vector<u8> >numArr[15]={
    {
        {1,1,1,1,1},
        {1,0,0,0,1},
        {1,1,1,1,1},
    },
    {
        {1,0,0,1,0},
        {1,1,1,1,1},
        {1,0,0,0,0},
    },
    {
        {1,1,1,0,1},
        {1,0,1,0,1},
        {1,0,1,1,1},
    },
    {
        {1,0,0,0,1},
        {1,0,1,0,1},
        {1,1,1,1,1},
    },
    {
        {0,0,1,1,1},
        {0,0,1,0,0},
        {1,1,1,1,1},
    },
    {
        {1,0,1,1,1},
        {1,0,1,0,1},
        {1,1,1,0,1},
    },
    {
        {1,1,1,1,1},
        {1,0,1,0,1},
        {1,1,1,0,1},
    },
    {
        {0,0,0,1,1},
        {0,0,0,0,1},
        {1,1,1,1,1},
    },
    {
        {1,1,1,1,1},
        {1,0,1,0,1},
        {1,1,1,1,1},
    },
    {
        {0,0,1,1,1},
        {0,0,1,0,1},
        {1,1,1,1,1},
    }
};

const std::vector<std::vector<u8> >sign[5]={
    {
        {0,0,1,0,0},
        {0,1,1,1,0},
        {0,0,1,0,0},
    },
    {
        {0,0,1,0,0},
        {0,0,1,0,0},
        {0,0,1,0,0},
    }
};

inline void writeSingleNumber(bmp &A,int num,const int x0,int const y0,const int sz=1,u8 r=255,u8 g=255,u8 b=255){
    int x=x0,y=y0;
    for(int i=0;i<3;i++,x+=sz){
        y=y0;
        for(int j=0;j<5;j++,y+=sz){
            for(int k=0;k<sz&&x+k<A.red.n;k++){
                for(int t=0;t<sz&&y+t<A.red.m;t++){
                    A.red(x+k,y+t)=numArr[num][i][j]*r;
                    A.green(x+k,y+t)=numArr[num][i][j]*g;
                    A.blue(x+k,y+t)=numArr[num][i][j]*b;
                }
            }
        }
    }
}

inline void writeSign(bmp &A,int s,const int x0,int const y0,const int sz=1,u8 r=255,u8 g=255,u8 b=255){
    int x=x0,y=y0;
    for(int i=0;i<3;i++,x+=sz){
        y=y0;
        for(int j=0;j<5;j++,y+=sz){
            for(int k=0;k<sz&&x+k<A.red.n;k++){
                for(int t=0;t<sz&&y+t<A.red.m;t++){
                    A.red(x+k,y+t)=sign[s][i][j]*r;
                    A.green(x+k,y+t)=sign[s][i][j]*g;
                    A.blue(x+k,y+t)=sign[s][i][j]*b;
                }
            }
        }
    }
}

inline void writeFloatPoint(bmp &A,const int x0,int const y0,const int sz=1,u8 r=255,u8 g=255,u8 b=255){
    for(int k=0;k<sz&&x0+k<A.red.n;k++){
        for(int t=0;t<sz&&y0+t<A.red.m;t++){
            A.red(x0+k,y0+t)=r;
            A.green(x0+k,y0+t)=g;
            A.blue(x0+k,y0+t)=b;
        }
    }
}

inline void writeDouble(bmp &A,double num,const int x0,int const y0,const int sz=1,u8 r=255,u8 g=255,u8 b=255){
    int x=x0,y=y0;
    if(num<0){
        writeSign(A,1,x,y,sz,r,g,b);
        x+=sz*4;
        num*=-1;
    }
    std::string n=std::to_string(num);
    n=n.substr(0,n.find('.')+3);
    for(char c:n){
        if(c=='.'){
            writeFloatPoint(A,x,y,sz,r,g,b);
            x+=sz*2;
        }else{
            writeSingleNumber(A,c-'0',x,y,sz,r,g,b);
            x+=sz*4;
        }
    }
}

inline void writeInt(bmp &A,int num,const int x0,int const y0,const int sz=1,u8 r=255,u8 g=255,u8 b=255){
    int x=x0,y=y0;
    if(num<0){
        writeSign(A,1,x,y,sz,r,g,b);
        x+=sz*4;
        num*=-1;
    }
    std::string n=std::to_string(num);
    for(char c:n){
        writeSingleNumber(A,c-'0',x,y,sz,r,g,b);
        x+=sz*4;
    }
}

}

#endif /* writeNumber_h */
