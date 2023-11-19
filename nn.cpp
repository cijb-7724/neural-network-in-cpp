/*
入力(p, q)が
中心原点，半径３の円の
内側にあるか外側にあるかを判定する
ニューラルネットワーク
*/
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <set>
#include <math.h>
#include <string>
#include <random>

using namespace std;

#define pi 3.14159265358979323846
#define yes "Yes"
#define no "No"
#define yesno(bool) if(bool){cout<<"Yes"<<endl;}else{cout<<"No"<<endl;}

#define tp() cout << "here~~" << endl


//型エイリアス vector<set<pair<tuple : bool<char<string<int<ll<ull
using ll = long long;
using ull = unsigned long long;
using vb = vector<bool>;
using vc = vector<char>;
using vs = vector<string>;
using vi = vector<int>;
using vll = vector<long long>;
using vd = vector<double>;
using si = set<int>;
using sll = set<ll>;
using msi = multiset<int>;
using msll = multiset<ll>;
using mss = multiset<string>;
using pii = pair<int, int>;
using pill = pair<int, ll>;
using plli = pair<ll, int>;
using pllll = pair<long long, long long>;
using vvb = vector<vector<bool>>;
using vvc = vector<vector<char>>;
using vvs = vector<vector<string>>;
using vvi = vector<vector<int>>;
using vvll = vector<vector<ll>>;
using vvd = vector<vector<double>>;
using vsi = vector<set<int>>;
using vpii = vector<pair<int, int>>;
using vpllll = vector<pair<long long, long long>>;
using spii = set<pair<int, int>>;


double h_sigmoid(double x) {
    return 1/(1+exp(-x));
}
double h_tash(double x) {
    return (exp(x)-exp(-x)) / (exp(x)+exp(-x));
}
double h_ReLU(double x) {
    return (x > 0) ? x : 0;
}
void h_ReLUMatrix(vvd &x) {
    int n = x.size(), m = x[0].size();
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            x[i][j] = h_ReLU(x[i][j]);
        }
    }
}
void showMatrix(vvd &a) {
    int n = a.size(), m = a[0].size();
    for (int j=0; j<m; ++j) cout << "--";
    cout << endl;
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            cout << a[i][j] << ' ';
        }
        cout << endl;
    }
    for (int j=0; j<m; ++j) cout << "--";
    cout << endl;
}
vvd softMax(vvd &x) {
    int n = x.size();
    int m = x[0].size();
    vvd y = x;
    for (int i=0; i<n; ++i) {
        double mx = *max_element(y[i].begin(), y[i].end());
        double deno = 0;
        for (int j=0; j<m; ++j) {
            y[i][j] -= mx;
            deno += exp(y[i][j]);
        }
        for (int j=0; j<m; ++j) {
            y[i][j] = exp(y[i][j]) / deno;
        }
    }
    return y;
}
double crossEntropy(vvd &y, vvd &t) {
    int n = y.size(), m = y[0].size();
    double sum = 0;
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            sum += t[i][j] * log(y[i][j]);
        }
    }
    return -sum/n;
}
// c = a * b
void multiMatrix(vvd &c, vvd &a, vvd &b) {
    int n = a.size(), m = b.size(), l = b[0].size();
    c.assign(n, vd(l, 0));
    for (int i=0; i<n; ++i) {
        for (int j=0; j<l; ++j) {
            for (int k=0; k<m; ++k) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}
// c = a .* b admal
void admMultiMatrix(vvd &c, vvd &a, vvd &b) {
    int n = a.size(), m = a[0].size();
    c.assign(n, vd(m));
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            c[i][j] = a[i][j] * b[i][j];
        }
    }
}
// c = a + b
void addMatrix(vvd& c, vvd &a, vvd &b) {
    if (a.size() != b.size() || a[0].size() != b[0].size()) {
        cout << "The matrix sizes are different." << endl;
        return;
    }
    int n = a.size(), m = a[0].size();
    c.assign(n, vd(m, 0));
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            c[i][j] += a[i][j] + b[i][j];
        }
    }
}
void tMatrix(vvd &a) {
    int n = a.size(), m = a[0].size();
    vvd t = a;
    a.assign(m, vd(n));
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            a[j][i] = t[i][j];
        }
    }
}
bool judgeTerm(double x, double y){ return (x*x + y*y < 9) ? true : false;}
void makeData(vvd &x, int n, int seed=0) {
    //条件を満たす点と満たさない点をｎ個ずつ作る
    //rd()は実行毎に異なる
    //rand()は実行毎に同じ
    random_device rd;
    // mt19937 gen(rd());//random
    mt19937 gen(seed);//seed
    uniform_real_distribution<> dist(-6, 6);
    x.assign(2*n, vd(2, 0));
    int id = 0;
    while(id < n) {
        double a, b;
        a = dist(gen);
        b = dist(gen);
        if (judgeTerm(a, b)) {
            x[id][0] = a;
            x[id][1] = b;
            ++id;
        }
    }
    while(id < n*2) {
        double a, b;
        a = dist(gen);
        b = dist(gen);
        if (!judgeTerm(a, b)) {
            x[id][0] = a;
            x[id][1] = b;
            ++id;
        }
    }
}

vvd calc_r_hL_x3(vvd &x, vvd &t) {
    int n = x.size(), m = x[0].size();
    double ips = 0.01, fLup, fLdown;
    vvd tmp(n, vd(m, 0));
    for (int s=0; s<n; ++s) {
        for (int j=0; j<m; ++j) {
            tmp[s][j] = -t[s][j] / x[s][j];
            for (int k=0; k<m; ++k) {
                if (j == k) tmp[s][j] -= t[s][j] / x[s][j];
                else tmp[s][j] += t[s][k] / x[s][k];
            }
            tmp[s][j] /= n;
        }
    }
    return tmp;
}
vvd calc_r_h3_a3 (vvd &a, vvd &x) {
    int n = a.size(), m = a[0].size();
    vvd tmp(n, vd(m, 0));
    for (int s=0; s<n; ++s) {
        for (int j=0; j<m; ++j) {
            tmp[s][j] = x[s][j]*(1-x[s][j]);
        }
    }
    return tmp;
}

vvd calc_r_h2_a2 (vvd &a, vvd &x) {
    int n = a.size(), m = a[0].size();
    vvd tmp(n, vd(m, 0));
    for (int s=0; s<n; ++s) {
        for (int j=0; j<m; ++j) {
            if (a[s][j] > 0) tmp[s][j] = 1;
        }
    }
    return tmp;
}

void updateWeights(vvd &w, vvd &rw, double eta) {
    if (!(w.size() == rw.size() && w[0].size() == rw[0].size())) {
        cout << "the sizes are different" << endl;
    }
    int n = w.size(), m = w[0].size();
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            w[i][j] -= eta * rw[i][j];
        }
    }
}


int main() {
    vvd x0, x1, x2, x3, a1, a2, a3, w1, w2, w3;
    
    w1 = {{-0.35, -0.52, -0.96}, {-0.88, -0.76, -0.086}};
    w2 = {{-0.47, -0.32, 0.93}, {0.47, -0.015, 0.88}, {-0.13, -0.22, 1.1}};
    w3 = {{-1.2, 0.47}, {0.16, 0.75}, {-1.8, -0.85}};
    int n = 10;


    //教師データ{1,0}内側
    vvd t;
    // t = {{0, 1}};
    vd tmp = {1, 0};//inner
    for (int i=0; i<n/2; ++i) t.push_back(tmp);
    tmp = {0, 1};//outer
    for (int i=0; i<n/2; ++i) t.push_back(tmp);
    

    makeData(x0, n/2);
    // x0 = {{2, 3}};
    cout << "x0 = " << endl;
    showMatrix(x0);
    cout << "w1 = " << endl;
    showMatrix(w1);
    cout << "w2 = " << endl;
    showMatrix(w2);
    cout << "w3 = " << endl;
    showMatrix(w3);

    multiMatrix(a1, x0, w1);//a1 = w1 * x0

    cout << "a1 = " << endl;
    showMatrix(a1);

    h_ReLUMatrix(a1);
    x1 = a1;

    cout << "x1 = " << endl;
    showMatrix(x1);

    multiMatrix(a2, x1, w2);//a2 = w2 * x1

    cout << "a2 = " << endl;
    showMatrix(a2);

    h_ReLUMatrix(a2);
    x2 = a2;

    cout << "x2 = " << endl;
    showMatrix(a2);

    multiMatrix(a3, x2, w3);//a1 = w1 * x0

    cout << "a3 = " << endl;
    showMatrix(a3);

    x3 = softMax(a3);

    cout << "x3 = " << endl;
    showMatrix(x3);

    cout << "teacher = " << endl;
    showMatrix(t);

    cout << "cross entropy ";
    cout << crossEntropy(x3, t) << endl;

    //back propagation
    vvd r_hL_x3 = calc_r_hL_x3(x3, t);
    vvd r_h3_a3 = calc_r_h3_a3(a3, x3);
    
    cout << "hL / x3" << endl;
    showMatrix(r_hL_x3);
    cout << "h3 / a3" << endl;
    showMatrix(r_h3_a3);

    vvd Delta3;
    admMultiMatrix(Delta3, r_hL_x3, r_h3_a3);
    cout << "del3" << endl;
    showMatrix(Delta3);

    vvd r_L_w3, tx2 = x2;
    tMatrix(tx2);

    cout << "t x2" << endl;
    showMatrix(tx2);

    multiMatrix(r_L_w3, tx2, Delta3);
    cout << "L / W3" << endl;
    showMatrix(r_L_w3);
    //ok


    vvd r_h2_a2 = calc_r_h2_a2(a2, x2);
    cout << "h2 / a2" << endl;
    showMatrix(r_h2_a2);

    vvd tw3 = w3;
    tMatrix(tw3);
    cout << "t w3" << endl;
    showMatrix(tw3);

    vvd tmp2;
    multiMatrix(tmp2, Delta3, tw3);

    vvd Delta2;
    admMultiMatrix(Delta2, r_h2_a2, tmp2);
    cout << "del2" << endl;
    showMatrix(Delta2);

    vvd tx1 = x1;
    tMatrix(tx1);
    cout << "t x1" << endl;
    showMatrix(tx1);

    vvd r_L_w2;
    multiMatrix(r_L_w2, tx1, Delta2);
    cout << "L / W2" << endl;
    showMatrix(r_L_w2);
    //ok

    vvd r_h1_a1 = calc_r_h2_a2(a1, x1);
    cout << "h1 / a1" << endl;
    showMatrix(r_h1_a1);

    vvd tw2 = w2;
    tMatrix(tw2);
    cout << "t w2" << endl;
    showMatrix(tw2);

    vvd tmp3;
    multiMatrix(tmp3, Delta2, tw2);

    vvd Delta1;
    admMultiMatrix(Delta1, r_h1_a1, tmp3);
    cout << "del1" << endl;
    showMatrix(Delta1);

    vvd tx0 = x0;
    tMatrix(tx0);
    cout << "t x0" << endl;
    showMatrix(tx0);

    vvd r_L_w1;
    multiMatrix(r_L_w1, tx0, Delta1);
    cout << "L / W1" << endl;
    showMatrix(r_L_w1);
    //ok?


    



}


/*

*/


 