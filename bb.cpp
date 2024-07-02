#include<bits/stdc++.h>
using namespace std;

const int N=110;
//节点信息
typedef struct Node{
    int machine=-1;
    int jobID=-1;
    int w=-1;
    bool isflag=false;
}Node;
Node Node_list[N];
int Node_idx=1;
vector<vector<int>> machine_list(5,vector<int>());
int state[N];
//图结构信息
int h1[N], e1[N], ne1[N], idx1, p1[N];//正向图信息
int h2[N], e2[N], ne2[N], idx2, p2[N];//反向图信息
int ht1[N], et1[N], net1[N], idxt1, pt1[N];//正向图信息
int ht2[N], et2[N], net2[N], idxt2, pt2[N];//反向图信息

//图节点到终节点的最长距离
int dist[N];

//图节点到源节点的最长距离
int dist_s[N];

//作业信息
int n=0;//作业个数
int sn=0;//节点个数


void add(int a,int b,int p){
    e1[idx1]=b,ne1[idx1]=h1[a],p1[idx1]=p,h1[a]=idx1++;
    e2[idx2]=a,ne2[idx2]=h2[b],p2[idx2]=p,h2[b]=idx2++;
}

void add_temp(int a,int b,int p){
    et1[idxt1]=b,net1[idxt1]=ht1[a],pt1[idxt1]=p,ht1[a]=idxt1++;
    et2[idxt2]=a,net2[idxt2]=ht2[b],pt2[idxt2]=p,ht2[b]=idxt2++;
}

void copy_graph(){
    for(int i=0;i<N;i++){
        ht1[i]=h1[i];
        ht2[i]=h2[i];
        et1[i]=e1[i];
        et2[i]=e2[i];
        net1[i]=ne1[i];
        net2[i]=ne2[i];
        pt1[i]=p1[i];
        pt2[i]=p2[i];
    }
    idxt1=idx1;
    idxt2=idx2;
}

void init_nodelist(){

    // printf("please input the number of jobs:\n");
    cin>>n>>sn;
    for(int i=0;i<n;i++){
        int nn;
        // printf("please input the number of stages\n");
        cin>>nn;
        for(int j=0;j<nn;j++){
            if(j==0){
                add(0,Node_idx,0);
            }else{
                add(Node_idx-1,Node_idx,Node_list[Node_idx-1].w);
            }
            int m,jobID,w;
            cin>>m>>jobID>>w;
            Node node=Node();
            node.machine=m;
            node.jobID=jobID;
            node.w=w;
            Node_list[Node_idx]=node;
            machine_list[m].push_back(Node_idx);
            Node_idx++;
        }
        add(Node_idx-1,sn-1,Node_list[Node_idx-1].w);
    }
}

void compute_dist_iter(){
    int sink=sn-1;
    dist[sink]=0;
    stack<int> st;
    st.push(sink);
    while(!st.empty()){
        int root=st.top();
        st.pop();
        for(int i=ht2[root];i!=-1;i=net2[i]){
            int temp=et2[i];
            bool flag=true;
            int mx=0;
            for(int j=ht1[temp];j!=-1;j=net1[j]){
                int ttemp=et1[j];
                if(dist[ttemp]==-1){
                    flag=false;
                    break;
                }
                mx=max(dist[ttemp],mx);
            }
            if(flag){
                dist[temp]=mx+pt2[i];
                st.push(temp);
            }
        }
    }
    
}

void compute_dist_s_iter(){
    int source=0;
    dist_s[source]=0;
    stack<int> st;
    st.push(source);
    while(!st.empty()){
        int root=st.top();
        st.pop();
        for(int i=ht1[root];i!=-1;i=net1[i]){
            int temp=et1[i];
            bool flag=true;
            //int mx=0;
            for(int j=ht2[temp];j!=-1;j=net2[j]){
                int ttemp=et2[j];
                if(dist_s[ttemp]==-1){
                    flag=false;
                    break;
                }
            }
            if(flag){
                for(int j=ht2[temp];j!=-1;j=net2[j]){
                    int ttemp=et2[j];
                    dist_s[temp]=max(dist_s[ttemp]+pt2[j],dist_s[temp]);
                }
                st.push(temp);
            }
        }
    }
    
}

int rj_max_1(vector<vector<int>> prd_mix){
    int num_j=prd_mix[0].size();
    unordered_set<int> no_use;
    vector<int> path;
    int cur_time=0;
    stack<int> st_j;
    for(int i=0;i<num_j;i++){
        st_j.push(i);
        no_use.emplace(i);
    } 
    while(!st_j.empty()){
        if(st_j.size()==1){
            int tt=st_j.top();
            no_use.erase(tt);
            path.push_back(tt);
            break;
        }
        int proper_root=-1;
        int lower_bound=999999;
        for(int i=0;i<int(st_j.size());i++){
            int l_m=-999999;
            int temp=st_j.top();
            st_j.pop();
            int vir_cur_time=max(cur_time,prd_mix[1][temp])+prd_mix[0][temp];
            if(vir_cur_time-prd_mix[2][temp]>l_m) l_m=vir_cur_time-prd_mix[2][temp];
            for(auto vir:no_use){
                if(vir!=temp){
                    vir_cur_time=max(vir_cur_time,prd_mix[1][vir])+prd_mix[0][vir];
                    if(vir_cur_time-prd_mix[2][vir]>l_m) {
                        l_m=vir_cur_time-prd_mix[2][vir];
                    }
                }
            }
            if(l_m<lower_bound){
                lower_bound=l_m;
                proper_root=temp;
            }
        }
        path.push_back(proper_root);
        cur_time=max(cur_time,prd_mix[1][proper_root])+prd_mix[0][proper_root];
        no_use.erase(proper_root);
        for(auto vir:no_use){
            st_j.push(vir);
        }
    }
    cur_time=0;
    int l_max=-999999;
    
    for(int i=0;i<int(path.size()-1);i++){
        int vv=path[i];
        
        cur_time=max(cur_time,prd_mix[1][vv])+prd_mix[0][vv];
        // cout<<"vv:"<<vv<<" "<<prd_mix[0][vv]<<" "<<prd_mix[1][vv]<<" "<<prd_mix[2][vv]<<endl;
        if(cur_time-prd_mix[2][vv]>l_max) {
            l_max=cur_time-prd_mix[2][vv];
        }
    }
    // cout<<"lm:"<<l_max<<endl;
    return l_max;
}

vector<int> Schedule(){
    vector<int> path;//调度策略
    // stack<int> st_root;
    // st_root.push(0);
    unordered_set<int> omg;
        // cout<<"path.size()"<<endl;
    for(int i=h1[0];i!=-1;i=ne1[i]){
            int next_follower=e1[i];
            omg.emplace(next_follower);
    }    
    cout<<"path.size()"<<endl;
    while(!omg.empty()){
        // int root=st_root.top();
        // st_root.pop();
        copy_graph();    
        memset(dist, -1, sizeof dist);
        memset(dist_s, -1, sizeof dist_s);
        compute_dist_s_iter();
        int star_i=0;
        int t_omg=999999;
        for(auto o:omg){
            
            int tt_omg=dist_s[o]+Node_list[o].w;
            if(tt_omg<t_omg){
                t_omg=tt_omg;
                star_i=Node_list[o].machine;
            }
        }
        vector<int> omg_pie;
        for(auto o:omg){
            if(Node_list[o].machine==star_i&&dist_s[o]<t_omg){
                omg_pie.push_back(o);
            }
        }
        int proper_root=0;
        int lower_bound=999999;
		
        for(auto branch:omg_pie){
            copy_graph();
            //对复制图加边操作
            for(auto target:machine_list[star_i]){
                if(branch!=target&&Node_list[target].isflag==false){
                    add_temp(branch,target,Node_list[branch].w);
                }
            }
            memset(dist, -1, sizeof dist);
            memset(dist_s, -1, sizeof dist_s);
            compute_dist_iter();
            compute_dist_s_iter();
            int r[N],d[N];
            int makespan=dist[8];
            cout<<makespan<<endl;
            for(int i=1;i<sn-1;i++){
                r[i]=dist_s[i];
                d[i]=makespan-dist[i]+Node_list[i].w;
            }
            int L_max=-999999;
            for(int i=1;i<=4;i++){
                vector<vector<int>> prd_mix(3,vector<int>());
                for(auto nid:machine_list[i]){
                    prd_mix[0].push_back(Node_list[nid].w);
                    prd_mix[1].push_back(r[nid]);
                    prd_mix[2].push_back(d[nid]);
                }
                int l_m=rj_max_1(prd_mix);
                if(l_m>L_max){
                    L_max=l_m;
                }
            }
            if(makespan+L_max<lower_bound){
                proper_root=branch;
                lower_bound=makespan+L_max;
            }
        }
        // cout<<lower_bound<<endl;
        //完成节点确认，对真实图加边操作
        for(int i=h1[proper_root];i!=-1;i=ne1[i]){
            if(Node_list[e1[i]].isflag==false&&e1[i]!=sn-1){
                omg.emplace(e1[i]);
            }
        }
        for(auto target:machine_list[star_i]){
            if(proper_root!=target){
                add(proper_root,target,Node_list[proper_root].w);
            }
        }
        Node_list[proper_root].isflag=true;
        omg.erase(proper_root);
        path.push_back(proper_root);
    }
    return path;
}

void test(vector<int> path){
    for(int i=0;i<path.size();i++){
        int vv=path[i];
        cout<<Node_list[vv].machine<<" "<<Node_list[vv].jobID<<endl;
    }
}



int main(){

    memset(h1, -1, sizeof h1);
    memset(h2, -1, sizeof h2);
    memset(ht1, -1, sizeof ht1);
    memset(ht2, -1, sizeof ht2);
    memset(dist, -1, sizeof dist);
    memset(dist_s, -1, sizeof dist_s);
    
    init_nodelist();
    vector<int> path=Schedule();

    test(path);
    return 0;
}

