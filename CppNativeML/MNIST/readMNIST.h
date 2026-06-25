//
//  readMNIST.h
//  CSE302
//
//  Created by Sora Sugiyama on 3/24/25.
//  Modified the code of https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c Answer by Jayhello.
//

#ifndef readMNIST_h
#define readMNIST_h

#include <iostream>
#include <fstream>
#include <cstddef>
#include <vector>
#include <string>
#include <cassert>
#include <memory>

namespace MNIST{

using u8=u_int8_t;
using u32=u_int32_t;
using mpix=std::vector<char>;

namespace{

void swap_endian(u32 &val){
    val=((val<<8)&0xFF00FF00)|((val>>8)&0xFF00FF);
    val=(val<<16)|(val>>16);
}

}

inline std::vector<std::pair<mpix,u32> >readMNIST(const std::string img_path,const std::string lab_path,int n0=-1){
    std::ifstream img_file(img_path.c_str(),std::ios::in|std::ios::binary);
    std::ifstream lab_file(lab_path.c_str(),std::ios::in|std::ios::binary);
    assert(("Cannot open image file — check kMNISTDir path in main.cpp\n" && img_file.is_open()));
    assert(("Cannot open label file — check kMNISTDir path in main.cpp\n" && lab_file.is_open()));
    u32 magic=0,n_img=0,n_lab=0,r=0,c=0;
    
    img_file.read(reinterpret_cast<char*>(&magic),4);
    swap_endian(magic);
    assert(("Incorrect image file magic number.\n"&&magic==2051));
    
    lab_file.read(reinterpret_cast<char*>(&magic),4);
    swap_endian(magic);
    assert(("Incorrect label file magic number.\n"&&magic==2049));
    
    img_file.read(reinterpret_cast<char*>(&n_img),4);
    swap_endian(n_img);
    lab_file.read(reinterpret_cast<char*>(&n_lab),4);
    swap_endian(n_lab);
    assert(("The number of image and the number label are must be same.\n"&&n_img==n_lab));
    if(n0==-1)n0=n_img;
    
    img_file.read(reinterpret_cast<char*>(&r),4);
    swap_endian(r);
    img_file.read(reinterpret_cast<char*>(&c),4);
    swap_endian(c);
    
    
    std::vector<std::pair<mpix,u32> >ret(n0);
    
    for(u32 i=0;i<n0;i++){
        mpix &pixels=ret[i].first;
        char lab='0';
        pixels.resize(r*c);
        img_file.read(pixels.data(),r*c);
        lab_file.read(&lab,1);
        ret[i].second=u32(lab);
    }
    
    img_file.close();
    lab_file.close();
    
    return ret;
}


}

#endif /* readMNIST_h */
