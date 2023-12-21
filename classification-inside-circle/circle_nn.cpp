/*
入力(p, q)が
中心原点，半径３の円の
内側にあるか外側にあるかを判定する
ニューラルネットワーク
*/

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <fstream>//
#include <sstream>//iranaikamo

using namespace std;
using vd = vector<double>;
using vvd = vector<vector<double>>;


vvd multiMatrix(const vvd &a, const vvd &b);        // c + a * b
vvd admMultiMatrix(const vvd &a, const vvd &b);     // c = a .* b (admal)
vvd addMatrix(const vvd &a, const vvd &b);          // c = a + b
vvd tMatrix(const vvd &a);                          // a = a^T

typedef struct {
    vvd w;
    vvd b;
} layer_t;

random_device rd;
long long SEED = 0;//実行毎に同じ乱数生成
// long long seed = rd();//実行毎に異なる
mt19937 engine(SEED);
uniform_real_distribution<> distCircle(-6, 6);


double gaussianDistribution (double mu, double sig) {
    normal_distribution <> dist(mu, sig);
    return dist(engine);
}
double h_sigmoid(double x) {
    return 1/(1+exp(-x));
}
double h_tash(double x) {
    return (exp(x)-exp(-x)) / (exp(x)+exp(-x));
}
double h_ReLU(double x) {
    return (x > 0) ? x : 0;
}

vvd h_ReLUMatrix(vvd &x) {
    int n = x.size(), m = x[0].size();
    vvd tmp(n, vd(m));
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            tmp[i][j] = h_ReLU(x[i][j]);
        }
    }
    return tmp;
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

void showMatrixB(vvd &a) {
    int m = a[0].size();
    for (int j=0; j<m; ++j) cout << "--";
    cout << endl;

    for (int j=0; j<m; ++j) cout << a[0][j] << ' ';
    cout << endl;

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
            if (y[i][j] == 0) {
                cout << "log 0!!!!!!!!!!!!!!!!" << endl;
            } else if (y[i][j] < 0) {
                cout << "log -x !!!!!!!!!!!!!!!!" << endl;
            }
            // if (y[i][j] <= 1e-5) y[i][j] = 1e-5;
            if (y[i][j] <= 0) y[i][j] = 1e-5;
            if (t[i][j]) sum += t[i][j] * log(y[i][j]);
        }
    }
    if (sum != sum) cout << "sum is nanananananananan" << endl;
    return -sum/n;
}

bool judgeTerm(double x, double y){ return (x*x + y*y < 9) ? true : false;}
//条件を満たす点と満たさない点をn/2個ずつ作る
vvd makeData(int n, int seed=0) {
    SEED = seed;
    vvd x;
    x.assign(n, vd(2, 0));
    int id = 0;
    while(id < n/2) {
        double a, b;
        a = distCircle(engine);
        b = distCircle(engine);
        if (judgeTerm(a, b)) {
            x[id][0] = a;
            x[id][1] = b;
            ++id;
        }
    }
    while(id < n) {
        double a, b;
        a = distCircle(engine);
        b = distCircle(engine);
        if (!judgeTerm(a, b)) {
            x[id][0] = a;
            x[id][1] = b;
            ++id;
        }
    }
    return x;
}
void makeInitialValue(vvd &table, double mu, double sig) {
    int n = table.size(), m = table[0].size();
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            table[i][j] = gaussianDistribution(mu, sig);
        }
    }
}

void expansionBias(vvd &b, int batch) {
    vd tmp = b[0];
    if (b.size() != 1) {
        cout << "bias batch size error" << endl;
        return ;
    }
    for (int i=0; i<batch-1; ++i) {
        b.push_back(tmp);
    }
}

vvd calc_r_hL_x3(vvd &x, vvd &t) {
    int n = x.size(), m = x[0].size();
    // double ips = 0.01, fLup, fLdown;
    vvd tmp(n, vd(m, 0));
    for (int s=0; s<n; ++s) {
        for (int j=0; j<m; ++j) {
            // tmp[s][j] = -t[s][j] / x[s][j];
            for (int k=0; k<m; ++k) {
                if (j == k) tmp[s][j] -= t[s][j] / x[s][j];
                else tmp[s][j] += t[s][k] / (x[s][k]);
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
            if (a[s][j] >= 0) tmp[s][j] = 1;
        }
    }
    return tmp;
}

void calc_r_L_b (vvd &rb, vvd &b, vvd &delta) {
    int n = b.size(), m = b[0].size();
    if (n != delta.size() || m != delta[0].size()) {
        cout << "size is not match" << endl;
    }
    rb.assign(1, vd(m, 0));
    for (int j=0; j<m; ++j) {
        for (int i=0; i<n; ++i) {
            rb[0][j] += delta[i][j];
        }
    }
    expansionBias(rb, n);
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
bool test = false;
double calcAccuracyRate(vvd &y, vvd &t) {
    int n = y.size();
    double sum = 0;
    for (int i=0; i<n; ++i) {
        if (y[i][0] < y[i][1] && t[i][1] || y[i][0] > y[i][1] && t[i][0]) sum += 1;
        else {
            // cout << "error instance" << endl;
            // cout << y[i][0] << ' ' << y[i][1] << endl;
            // cout << t[i][0] << ' ' << t[i][1] << endl;
        }
    }
    return sum / n;
}

void shuffleVVD(vvd &v, vector<int> &id) {
    vvd tmp = v;
    int n = v.size();
    for (int i=0; i<n; ++i) {
        for (int j=0; j<v[0].size(); ++j) {
            tmp[i][j] = v[id[i]][j];
        }
    }
    v = tmp;
}


int main() {
    vvd x0, x1, x2, x3, a1, a2, a3, w1, w2, w3, b1, b2, b3;
    vvd tmp1, r_hL_x3, r_h3_a3, Delta3, r_L_w3, tx2, r_h2_a2, tw3, tmp2, Delta2, tx1, r_L_w2, r_h1_a1, tw2, tmp3, Delta1, tx0, r_L_w1, r_L_b1, r_L_b2, r_L_b3;
    double eta = 0.1, attenuation = 0.7;
    int n = 1000;
    int loop = 6500;
    int batch_size = 1;
    vector<int> nn_form = {2, 3, 3, 2};
    int depth = nn_form.size()-1;
    vector<layer_t> nn(depth);

    vector<int> id(n);
    for (int i=0; i<n; ++i) id[i] = i;
    
    //Heの初期化
    for (int i=0; i<depth; ++i) {
        nn[i].w.assign(nn_form[i], vd(nn_form[i+1], 0));
        nn[i].b.assign(batch_size, vd(nn_form[i+1], 0));
        makeInitialValue(nn[i].w, 0, sqrt(2.0/nn_form[i]));
        makeInitialValue(nn[i].b, 0, sqrt(2.0/nn_form[i]));
        expansionBias(nn[i].b, 1);
    }
    
    //初期のパラメータ
    cout << "=========================================" << endl;
    cout << "first parameters" << endl;
    cout << "w" << endl;
    for (int i=0; i<depth; ++i) showMatrix(nn[i].w);
    cout << "b" << endl;
    for (int i=0; i<depth; ++i) showMatrixB(nn[i].b);
    cout << "=========================================" << endl;
    
    // return 0;
    //前半半分は円の内側，後半半分は円の外側
    x0 = makeData(n);
    //教師データの作成 {1,0}内側
    vvd t;
    for (int i=0; i<n/2; ++i) t.push_back({1, 0});//inside
    for (int i=0; i<n/2; ++i) t.push_back({0, 1});//outside
    
    //learn
    for (int i=0; i<loop; ++i) {
        //forward propagation
        shuffle(id.begin(), id.end(), engine);
        shuffleVVD(t, id);
        shuffleVVD(x0, id);

        vvd x00 = {{x0[0][0], x0[0][1]}};
        vvd t00 = {{t[0][0], t[0][1]}};

        //a1 = w1 * x0 + b1
        tmp1 = multiMatrix(x00, nn[0].w);
        a1 = addMatrix(tmp1, nn[0].b);
        x1 = h_ReLUMatrix(a1);
        //a2 = w2 * x1 + b2
        tmp1 = multiMatrix(x1, nn[1].w);
        a2 = addMatrix(tmp1, nn[1].b);
        x2 = h_ReLUMatrix(a2);
        //a3 = w3 * x2 + b3
        tmp1 = multiMatrix(x2, nn[2].w);
        a3 = addMatrix(tmp1, nn[2].b);
        x3 = softMax(a3);
        

        //たまに値の確認
        if (i % 500 == 0) {
            cout << i << " cross entropy ";
            cout << crossEntropy(x3, t00) << endl;
            cout << "accuracy rate ";
            cout << calcAccuracyRate(x3, t00) << endl;
        }

        //back propagation
        r_hL_x3 = calc_r_hL_x3(x3, t);
        r_h3_a3 = calc_r_h3_a3(a3, x3);
        Delta3 = admMultiMatrix(r_hL_x3, r_h3_a3);
        calc_r_L_b(r_L_b3, nn[2].b, Delta3);

        tx2 = tMatrix(x2);
        r_L_w3 = multiMatrix(tx2, Delta3);
        r_h2_a2 = calc_r_h2_a2(a2, x2);
        tw3 = tMatrix(nn[2].w);
        tmp2 = multiMatrix(Delta3, tw3);
        Delta2 = admMultiMatrix(r_h2_a2, tmp2);
        calc_r_L_b(r_L_b2, nn[1].b, Delta2);

        tx1 = tMatrix(x1);
        r_L_w2 = multiMatrix(tx1, Delta2);
        r_h1_a1 = calc_r_h2_a2(a1, x1);
        tw2 = tMatrix(nn[1].w);
        tmp3 = multiMatrix(Delta2, tw2);
        Delta1 = admMultiMatrix(r_h1_a1, tmp3);
        calc_r_L_b(r_L_b1, nn[0].b, Delta1);

        tx0 = tMatrix(x0);
        r_L_w1 = multiMatrix(tx0, Delta1);

        if ((i+1) % 800 == 0) eta *= attenuation;
        updateWeights(nn[0].w, r_L_w1, eta);
        updateWeights(nn[1].w, r_L_w2, eta);
        updateWeights(nn[2].w, r_L_w3, eta);
        updateWeights(nn[0].b, r_L_b1, eta);
        updateWeights(nn[1].b, r_L_b2, eta);
        updateWeights(nn[2].b, r_L_b3, eta);
        
    }


    // train set---------------------------------
    cout << "=========================================" << endl;
    cout << "train set rate" << endl;
    expansionBias(nn[0].b, n);
    expansionBias(nn[1].b, n);
    expansionBias(nn[2].b, n);


    //a1 = w1 * x0 + b1
    tmp1 = multiMatrix(x0, nn[0].w);
    a1 = addMatrix(tmp1, nn[0].b);
    x1 = h_ReLUMatrix(a1);
    //a2 = w2 * x1 + b2
    tmp1 = multiMatrix(x1, nn[1].w);
    a2 = addMatrix(tmp1, nn[1].b);
    x2 = h_ReLUMatrix(a2);
    //a3 = w3 * x2 + b3
    tmp1 = multiMatrix(x2, nn[2].w);
    a3 = addMatrix(tmp1, nn[2].b);
    x3 = softMax(a3);

    cout << "cross entropy";
    cout << crossEntropy(x3, t) << endl;
    cout << "accuracy rate ";
    cout << calcAccuracyRate(x3, t) << endl;

    // test set-------------------------------------
    cout << "=========================================" << endl;
    cout << "test" << endl;
    
    x0 = makeData(n);
    
    t.assign(0, vd(0));
    for (int i=0; i<n/2; ++i) t.push_back({1, 0});
    for (int i=0; i<n/2; ++i) t.push_back({0, 1});

    //a1 = w1 * x0 + b1
    tmp1 = multiMatrix(x0, nn[0].w);
    a1 = addMatrix(tmp1, nn[0].b);
    x1 = h_ReLUMatrix(a1);
    //a2 = w2 * x1 + b2
    tmp1 = multiMatrix(x1, nn[1].w);
    a2 = addMatrix(tmp1, nn[1].b);
    x2 = h_ReLUMatrix(a2);
    //a3 = w3 * x2 + b3
    tmp1 = multiMatrix(x2, nn[2].w);
    a3 = addMatrix(tmp1, nn[2].b);
    x3 = softMax(a3);

    test = true;

    cout << "cross entropy";
    cout << crossEntropy(x3, t) << endl;
    cout << "accuracy rate ";
    cout << calcAccuracyRate(x3, t) << endl;

    cout << "=========================================" << endl;
    cout << "w" << endl;
    for (int i=0; i<depth; ++i) showMatrix(nn[i].w);
    cout << "b" << endl;
    for (int i=0; i<depth; ++i) showMatrixB(nn[i].b);
    cout << "=========================================" << endl;
    
    
    

}


/*


*/

//  ##   ##    ##     ######   ######    ####    ##  ##
//  ### ###   ####    # ## #    ##  ##    ##     ##  ##
//  #######  ##  ##     ##      ##  ##    ##      ####
//  #######  ##  ##     ##      #####     ##       ##
//  ## # ##  ######     ##      ## ##     ##      ####
//  ##   ##  ##  ##     ##      ##  ##    ##     ##  ##
//  ##   ##  ##  ##    ####    #### ##   ####    ##  ##

// c = a * b
vvd multiMatrix(const vvd &a, const vvd &b) {
    int n = a.size(), m = b.size(), l = b[0].size();
    vvd c;
    c.assign(n, vd(l, 0));
    for (int i=0; i<n; ++i) {
        for (int j=0; j<l; ++j) {
            for (int k=0; k<m; ++k) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return c;
}
// c = a .* b (admal)
vvd admMultiMatrix(const vvd &a, const vvd &b) {
    int n = a.size(), m = a[0].size();
    vvd c;
    c.assign(n, vd(m));
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            c[i][j] = a[i][j] * b[i][j];
        }
    }
    return c;
}

// c = a + b
vvd addMatrix(const vvd &a, const vvd &b) {
    if (a.size() != b.size() || a[0].size() != b[0].size()) {
        cout << "The matrix sizes are different." << endl;
        vvd ret = {{0}};
        return ret;
    }
    int n = a.size(), m = a[0].size();
    vvd c;
    c.assign(n, vd(m, 0));
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            c[i][j] += a[i][j] + b[i][j];
        }
    }
    return c;
}

// a = a^T
vvd tMatrix(const vvd &a) {
    int n = a.size(), m = a[0].size();
    vvd t;
    t.assign(m, vd(n));
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            t[j][i] = a[i][j];
        }
    }
    return t;
}


//    ##       ####   ######    ####    ##   ##    ##     ######    ####     #####   ##   ##
//   ####     ##  ##  # ## #     ##     ##   ##   ####    # ## #     ##     ##   ##  ###  ##
//  ##  ##   ##         ##       ##      ## ##   ##  ##     ##       ##     ##   ##  #### ##
//  ##  ##   ##         ##       ##      ## ##   ##  ##     ##       ##     ##   ##  ## ####
//  ######   ##         ##       ##       ###    ######     ##       ##     ##   ##  ##  ###
//  ##  ##    ##  ##    ##       ##       ###    ##  ##     ##       ##     ##   ##  ##   ##
//  ##  ##     ####    ####     ####       #     ##  ##    ####     ####     #####   ##   ##





//  ######     ##       ####   ###  ##
//   ##  ##   ####     ##  ##   ##  ##
//   ##  ##  ##  ##   ##        ## ##
//   #####   ##  ##   ##        ####
//   ##  ##  ######   ##        ## ##
//   ##  ##  ##  ##    ##  ##   ##  ##
//  ######   ##  ##     ####   ###  ##

//  ######   ######    #####   ######     ##       ####     ##     ######    ####     #####   ##   ##
//   ##  ##   ##  ##  ##   ##   ##  ##   ####     ##  ##   ####    # ## #     ##     ##   ##  ###  ##
//   ##  ##   ##  ##  ##   ##   ##  ##  ##  ##   ##       ##  ##     ##       ##     ##   ##  #### ##
//   #####    #####   ##   ##   #####   ##  ##   ##       ##  ##     ##       ##     ##   ##  ## ####
//   ##       ## ##   ##   ##   ##      ######   ##  ###  ######     ##       ##     ##   ##  ##  ###
//   ##       ##  ##  ##   ##   ##      ##  ##    ##  ##  ##  ##     ##       ##     ##   ##  ##   ##
//  ####     #### ##   #####   ####     ##  ##     #####  ##  ##    ####     ####     #####   ##   ##


