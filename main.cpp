/// ============================================================================
/// Name        : Genetic-algorithm.cpp
/// Author      : Abd L-Rahman Sharaf
/// Email       : abdlrahman.sharaf@gmail.com
/// Copyright   : MIT lisence
/// Description : Hello World in C++, Ansi-style
/// ============================================================================

#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <string>
#include <map>
#include <vector>
#include <time.h>
#include <math.h>

using namespace std;

string max_gene, min_gene;
int n,m;
typedef vector<string> vec;
typedef map <int,string> ms;
typedef unsigned int ui;

#define print_vector(v) for(vec:: iterator it=v.begin(); it!=v.end(); ++it){cout<<*it<<endl;};
#define print_map(mp) for(ms::iterator it=mp.begin();it!=mp.end();it++){cout<<it->first<<" => "<<it->second<<endl;};


//Fitness function
int fitness(string str){    //Calculate how many 'ones' in between zeros like "010".
    int ctr=0;
    for(ui j=0; j<str.length(); j++){
        if(str[j-1]=='0' && str[j+1]=='0' && str[j]=='1')
            ctr++;
    }
    return ctr;
}

//Mutation function
string mutation(string gene){
    srand(time(NULL));
    int mut=(rand()%gene.length());
    if(gene[mut]=='1')
        gene[mut]='0';
    else
        gene[mut]='1';

    return gene;
}

//Crossover function
string* crossover(string male, string female){
    string* childs = new string[2];
    char gene1[male.size()];
    char gene2[female.size()];
    ui i,j;
//First child crossover
    for(i=0; i<(male.size()/2.0); i++){
        gene1[i]= male[i];
    }

    for(j=i; j<=female.size(); j++){
        gene1[j]=female[j];
    }
    childs[0]=gene1;
//Second child crossover
    for(i=0; i<(female.size()/2.0); i++){
        gene2[i]= female[i];
    }

    for(j=i; j<=male.size(); j++){
        gene2[j]=male[j];
    }
    childs[1]=gene2;

    return childs; //Returns the array of strings of the two childs as a pointer.
}


int main()
{
    vec v;
    ms fit_map;
    cout <<"Enter the number of the generations: \n";
    cin>>n;
    cout<<"Enter the number of the genes : \n";
    cin>>m;
    string gen[n][m];

    //Get the first population from the user as the first generation in the algorithm.
    cout<<"The generation 0 : "<<"\n\n";
    for(int l=0; l<m; l++){
        cout<<"Enter the "<<l<<" gene: "<<endl;
        cin>>gen[0][l];
        
        if(gen[0][l].length()<10){  //Make all genes with the same length of 10 bit in binary representation.
        string temp;
        for(int i=gen[0][l].length();i<10;i++)
        {
            temp=temp+"0";
        }
        gen[0][l]=temp+gen[0][l];
        }
        
        fit_map[fitness(gen[0][l])]= gen[0][l];
    }
    string worst;
    string best;
    worst=gen[0][0];
    best=gen[0][0];
    //Get worst and best gene for the first generation.
    for (int in=0; in<m; in++){
        if (fitness(worst)>fitness(gen[0][in])){
            worst= gen[0][in];
        }
        if(fitness(best)<fitness(gen[0][in])){
            best= gen[0][in];
        }
    }
    //Initialization for the best and worst gene in all generations.
    min_gene=worst;
    max_gene=best;

    //Start evolution next generations from here.
    for(int k=1; k<n; k++){
            int ctr=0, flag=0;
            string ml,fl;

            //Get the crossover elements by iterating through a sorted map and get the two middle elements in it.
            for (ms:: iterator it= fit_map.begin(); it!= fit_map.end(); it++){
                    if(ctr == floor(fit_map.size()/2.0) || ctr == floor(fit_map.size()/2.0)+1){
                        if(flag==0){
                            ml=(it->second);
                            flag++;
                            }
                        else{
                            fl=(it->second);
                            }
                    }
                    ctr++;
            }

            //Print all fitness evaluations in the map with its string.
            cout<<"\n Fitness evaluations => String gene \n";
            print_map(fit_map);
            fit_map.clear();

            //Start to make the genetics operations for the generation.
            ctr=0, flag=0;
            cout<<"Crossover elements are : "<<ml<<"\t"<<fl<<endl;
            string *st= crossover(ml,fl);
            string child1=(*st);
            st++;
            string child2=(*st);
            cout<<"Crossover 1st child : "<<child1<<"\t\t 2nd child : "<<child2<<endl;
            cout<<"The worst gene in the generation : "<<worst<<endl;
            cout<<"The best gene in the generation : "<<best<<endl;
            string mut_child=mutation(worst);
            cout<<"Mutation for the worst gene : "<<mut_child<<endl;
            gen[k][0]= mut_child;
            gen[k][1]= child1;
            gen[k][2]= child2;
            gen[k][3]= best;
            v.clear();

            //After clearing the vector pushing the new remainder genes into these vector for reproduction.
            for(int j=0; j<m; j++){
                    if((gen[k-1][j]!=worst) && (gen[k-1][j]!=best) && (gen[k-1][j]!=ml) && (gen[k-1][j]!=fl)){
                         v.push_back(gen[k-1][j]);
                    }
            }
            /*
            cout<<"\nVector Elements are :\n"; //If you want to check the remainder elements print all vector elements.
            print_vector(v);
            cout<<endl;
            */

            //Reproduction all the remainder genes into the following generation.
            for(ui j=0; j<v.size(); j++){
                    gen[k][j+4]=v.at(j);
            }

            //Print the genes in the same generation.
            cout<<"\t\t ********************************\n";
            cout<<"\t\t Generation "<<k<<" is : "<<endl;
            cout<<"\t\t ********************************\n";
            for(int f=0; f<m; f++){
                    cout<<"\t\t Gene "<<f<<" is : "<<gen[k][f]<<endl;
                    fit_map[fitness(gen[k][f])]= gen[k][f];
            }
            //Get the worst and the best genes in the generation.
            for (int in=0; in<m; in++){
                    if (fitness(worst)>fitness(gen[k][in])){
                        worst= gen[k][in];
                        }
            if(fitness(best)<fitness(gen[k][in])){
                    best= gen[k][in];
                    }
            }
            //Check every gene is the best gene came out of all the generations.
            if(fitness(max_gene)<fitness(best))
                max_gene=best;
            if(fitness(min_gene)>fitness(worst))
                min_gene=worst;
            char c;
            cout<<"\n\t\t\t\t Do you want to continue next generation ('Y' for continue or 'N' for stop)\n";
            cin>>c;
            if(c=='y' || c=='Y' )
                continue;
            if (c=='N' || c=='n')
                break;
    }
    //Print after all iterations the best and worst gene came out from all the generations.
    cout<<"\t\t The best gene through all generations is :   "<<max_gene<<"\t with fitness evaluation =  "<<fitness(max_gene)<<endl;
    cout<<"\t\t The worst gene through all generations is :   "<<min_gene<<"\t with fitness evaluation =  "<<fitness(min_gene)<<endl;

    return 0;
}
