//
//  GA.h
//  CSE302
//
//  Created by Sora Sugiyama on 3/22/25.
//

#ifndef GA_h
#define GA_h

#include <vector>
#include <random>
#include <functional>
#include <algorithm>

namespace optimizer{

inline std::random_device rd;

template<
class Dom,
class Codom,
class F,
class Mutate,
class Init,
class Comp=std::less<Codom>
>void GA(
        F objf,
        Mutate mut,
        Init init,
        Dom &coef,
        Comp comp=std::less<Codom>(),
        const int ng=39,int nG=10000){
    
    assert(("The Number of genes must be at least 3."&&ng>=3));
    
    std::vector<Dom>genes(39,coef);
    for(auto &gene:genes)init(gene);
    
    std::pair<Codom,Dom*>sup[5]={
        {objf(genes[0]),&genes[0]},
        {objf(genes[1]),&genes[1]},
        {objf(genes[2]),&genes[2]}
    };
    
    if(comp(sup[1].first,sup[0].first))swap(sup[0],sup[1]);
    if(comp(sup[2].first,sup[1].first))swap(sup[2],sup[1]);
    if(comp(sup[1].first,sup[0].first))swap(sup[0],sup[1]);
    
    auto sort=[&sup,&comp](void){
        if(comp(sup[2].first,sup[1].first))swap(sup[2],sup[1]);
        if(comp(sup[1].first,sup[0].first))swap(sup[0],sup[1]);
    };
    
    sort();
    
    while(nG--){
        for(int i=3;i<ng;i++){
            auto &gene=genes[i];
            std::pair<Codom,Dom*>cur={objf(gene),&gene};
            if(comp(cur.first,sup[2].first))sup[2]=cur;
            sort();
        }
        
        for(int i=0;i<ng;i++){
            auto &gene=genes[i];
            if((&gene==sup[0].second)||(&gene==sup[1].second)||(&gene==sup[2].second))continue;
            
            mut(gene,*sup[i%3].second);
        }
    }
    
    coef=*sup[0].second;
}

}
#endif /* GA_h */
