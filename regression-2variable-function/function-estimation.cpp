#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <fstream>
#include <sstream>

using namespace std;
using vd = vector<double>;
using vvd = vector<vector<double>>;
using vvvd = vector<vector<vector<double>>>;

typedef struct {
    vvd w;
    vvd b;
    vvd a;
    vvd x;
    vvd delta;
    vvd rw;
    vvd rb;
} layer_t;

//function
double judge_term(double x);
double true_function(double x, double y);
void make_data(vvd &x, vvd &t, int n);
void make_initial_value(vvd &table, double mu, double sig);
double calc_accuracy_rate(vvd &y, vvd &t);
void shuffle_VVD(vvd &v, vector<int> &id);
void outputfile(const vvd &output);
void drawing_by_python(vector<layer_t> &nn, int depth);
//MATRIX
void matrix_show(vvd &a);
void matrix_show_b(vvd &a);
vvd matrix_multi(const vvd &a, const vvd &b);
vvd matrix_adm_multi(const vvd &a, const vvd &b);
vvd matrix_adm_multi_tensor(const vvd &a, const vvvd &b);
vvd matrix_add(const vvd &a, const vvd &b);
vvd matrix_t(const vvd &a);
//ACTIVATION
double gaussianDistribution (double mu, double sig);
double h_sigmoid(double x);
double h_tash(double x);
double h_ReLU(double x);
double h_identity(double x);
vvd hm_ReLU(vvd &x);
vvd hm_softmax(vvd &x);
double hm_cross_entropy(vvd &y, vvd &t);
vvd hm_identity(vvd &x);
double hm_MSE(vvd &y, vvd &t);
vvd hm_tanh(vvd &x);

//BACK PROPAGATION
vvd expansion_bias(vvd &b, int batch);
vvd calc_r_cross_entropy(vvd &x, vvd &t);
vvvd calc_r_softmax(vvd &x);
vvd calc_r_ReLU (vvd &a);
vvd calc_r_bias (vvd &b, vvd &delta);
vvd calc_r_identity(vvd &x);
vvd calc_r_tanh(vvd &x);
vvd calc_r_MSE(vvd &y, vvd &t);
void updateWeights(vvd &w, vvd &rw, double eta);

random_device rd;
long long SEED = 0;//実行毎に同じ乱数生成
// long long SEED = rd();//実行毎に異なる乱数生成
mt19937 engine(SEED);
uniform_real_distribution<> distCircle(-10, 10);

//  ##   ##    ##      ####    ##   ##
//  ### ###   ####      ##     ###  ##
//  #######  ##  ##     ##     #### ##
//  #######  ##  ##     ##     ## ####
//  ## # ##  ######     ##     ##  ###
//  ##   ##  ##  ##     ##     ##   ##
//  ##   ##  ##  ##    ####    ##   ##

int main() {
    vvd x, t;
    double eta = 0.3, attenuation = 0.9;
    int n = 1000;
    int show_interval = 1000;
    int learning_plan = 1000;
    int loop = 20000;
    int batch_size = 30;
    vector<int> nn_form = {2, 6, 12, 8, 1};
    // vector<int> nn_form = {2, 4, 6, 8, 6, 1};
    
    int depth = nn_form.size()-1;

    vector<layer_t> nn(depth);

    vector<int> id(n);
    for (int i=0; i<n; ++i) id[i] = i;
    
    //Heの初期化
    for (int i=0; i<depth; ++i) {
        nn[i].w.assign(nn_form[i], vd(nn_form[i+1], 0));
        nn[i].b.assign(batch_size, vd(nn_form[i+1], 0));
        make_initial_value(nn[i].w, 0, sqrt(2.0/nn_form[i]));
        make_initial_value(nn[i].b, 0, sqrt(2.0/nn_form[i]));
        nn[i].b = expansion_bias(nn[i].b, batch_size);
    }
    
    //初期のパラメータの表示
    cout << "first parameters" << endl;
    // for (int i=0; i<40; ++i) cout << "=";
    // cout << endl;
    // for (int i=0; i<depth; ++i) {
    //     cout << "w " << i+1 << endl; 
    //     matrix_show(nn[i].w);
    // }
    // for (int i=0; i<depth; ++i) {
    //     cout << "b " << i+1 << endl;
    //     matrix_show_b(nn[i].b);
    // }
    // for (int i=0; i<40; ++i) cout << "=";
    // cout << endl;

    //訓練セットの作成
    make_data(x, t, n);

    //learn
    for (int i=0; i<loop; ++i) {
        //mini batchの作成
        vvd x0, t0;
        shuffle(id.begin(), id.end(), engine);
        shuffle_VVD(t, id);
        shuffle_VVD(x, id);
        //全データから先頭batchi_sizeだけ個mini batchを取得
        for (int j=0; j<batch_size; ++j) {
            x0.push_back(x[j]);
            t0.push_back(t[j]);
        }

        //forward propagation
        for (int k=0; k<depth; ++k) {
            if (k == 0) nn[k].a = matrix_add(matrix_multi(x0, nn[k].w), nn[k].b);
            else nn[k].a = matrix_add(matrix_multi(nn[k-1].x, nn[k].w), nn[k].b);
            if (k < depth-1) nn[k].x = hm_tanh(nn[k].a);
            else nn[k].x = hm_identity(nn[k].a);
        }
        
        //back propagation
        for (int k=depth-1; k>=0; --k) {
            if (k == depth-1) {
                vvd r_fL_xk;
                vvd r_hk_ak;
                r_fL_xk = calc_r_MSE(nn[k].x, t0);
                r_hk_ak = calc_r_identity(nn[k].x);
                nn[k].delta = matrix_adm_multi(r_fL_xk, r_hk_ak);
            } else {
                vvd r_h_a;
                r_h_a = calc_r_tanh(nn[k].a);
                nn[k].delta = matrix_adm_multi(r_h_a, matrix_multi(nn[k+1].delta, matrix_t(nn[k+1].w)));
            }
            
            nn[k].rb = calc_r_bias(nn[k].b, nn[k].delta);
            if (k != 0) nn[k].rw = matrix_multi(matrix_t(nn[k-1].x), nn[k].delta);
            else nn[k].rw = matrix_multi(matrix_t(x0), nn[k].delta);
        }

        //update parameters
        for (int k=0; k<depth; ++k) {
            updateWeights(nn[k].w, nn[k].rw, eta);
            updateWeights(nn[k].b, nn[k].rb, eta);
        }
        //学習率の更新
        if ((i+1) % learning_plan == 0) eta *= attenuation;

        //たまに性能の確認
        if (i % show_interval == 0) {
            cout << i << " MSE ";
            cout << hm_MSE(nn[depth-1].x, t0) << endl;
        }
    }

    // train set---------------------------------
    for (int i=0; i<40; ++i) cout << "=";
    cout << endl;
    cout << "train set" << endl;
    for (int i=0; i<depth; ++i) {
        nn[i].b = expansion_bias(nn[i].b, n);
    }
    //forward propagation
    for (int k=0; k<depth; ++k) {
        if (k == 0) nn[k].a = matrix_add(matrix_multi(x, nn[k].w), nn[k].b);
        else nn[k].a = matrix_add(matrix_multi(nn[k-1].x, nn[k].w), nn[k].b);
        if (k < depth-1) nn[k].x = hm_tanh(nn[k].a);
        else nn[k].x = hm_identity(nn[k].a);
    }
    cout << " MSE ";
    cout << hm_MSE(nn[depth-1].x, t) << endl;
    for (int i=0; i<40; ++i) cout << "=";
    cout << endl;

    // test set-------------------------------------
    for (int i=0; i<40; ++i) cout << "=";
    cout << endl;
    cout << "test set" << endl;
    //新しいデータをランダムに作成
    make_data(x, t, n);
    
    //test set は単に順伝播させて，正解率を見るだけだからシャッフルは必要ない

    //forward propagation
    for (int k=0; k<depth; ++k) {
        if (k == 0) nn[k].a = matrix_add(matrix_multi(x, nn[k].w), nn[k].b);
        else nn[k].a = matrix_add(matrix_multi(nn[k-1].x, nn[k].w), nn[k].b);
        if (k < depth-1) nn[k].x = hm_tanh(nn[k].a);
        else nn[k].x = hm_identity(nn[k].a);
    }
    cout << " MSE ";
    cout << hm_MSE(nn[depth-1].x, t) << endl;
    for (int i=0; i<40; ++i) cout << "=";
    cout << endl;

    //最後のパラメータの表示
    cout << "last parameters" << endl;
    // for (int i=0; i<40; ++i) cout << "=";
    // cout << endl;
    // for (int i=0; i<depth; ++i) {
    //     cout << "w " << i+1 << endl; 
    //     matrix_show(nn[i].w);
    // }
    // for (int i=0; i<depth; ++i) {
    //     cout << "b " << i+1 << endl;
    //     matrix_show_b(nn[i].b);
    // }
    // for (int i=0; i<40; ++i) cout << "=";
    // cout << endl;

    //一旦csvに出力したのちpythonで描画してみる
    drawing_by_python(nn, depth);
}

//  #######  ##   ##  ##   ##    ####   ######    ####     #####   ##   ##
//   ##   #  ##   ##  ###  ##   ##  ##  # ## #     ##     ##   ##  ###  ##
//   ## #    ##   ##  #### ##  ##         ##       ##     ##   ##  #### ##
//   ####    ##   ##  ## ####  ##         ##       ##     ##   ##  ## ####
//   ## #    ##   ##  ##  ###  ##         ##       ##     ##   ##  ##  ###
//   ##      ##   ##  ##   ##   ##  ##    ##       ##     ##   ##  ##   ##
//  ####      #####   ##   ##    ####    ####     ####     #####   ##   ##

// return 3*np.sin(np.sqrt(x**2 + y**2)/2)

double true_function(double x, double y) {
    double z = 0;
    z += 3*sin(sqrt(x*x+y*y)/2);
    // z += (x-5)*(x-5)/30;
    // z += (y+5)*(y+5)/30;
    return z;
}
//条件を満たす点と満たさない点をn/2個ずつ作る
void make_data(vvd &x, vvd &t, int n) {
    x.assign(n, vd(2));
    t.assign(n, vd(1));
    double a, b;
    for (int i=0; i<n; ++i) {
        a = distCircle(engine);
        b = distCircle(engine);
        x[i][0] = a;
        x[i][1] = b;
        t[i][0] = true_function(a, b);
    }
}
void make_initial_value(vvd &table, double mu, double sig) {
    int n = table.size(), m = table[0].size();
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            table[i][j] = gaussianDistribution(mu, sig);
        }
    }
}


double calc_accuracy_rate(vvd &y, vvd &t) {
    int n = y.size(), m = y[0].size();
    double sum = 0;
    for (int i=0; i<n; ++i) {
        double mx = *max_element(y[i].begin(), y[i].end());
        for (int j=0; j<m; ++j) {
            if (y[i][j] == mx && t[i][j]) sum += 1;
        }
    }
    return sum / n;
}

void shuffle_VVD(vvd &v, vector<int> &id) {
    vvd tmp = v;
    int n = v.size();
    for (int i=0; i<n; ++i) {
        for (int j=0; j<v[0].size(); ++j) {
            tmp[i][j] = v[id[i]][j];
        }
    }
    v = tmp;
}

//x, y, tを列挙
void outputfile(const vvd &output) {
    int n = output.size(), m = output[0].size();
    string fname = "point_.csv";
    ofstream outputFile (fname);

    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            outputFile << output[i][j];
            if (j != m-1) outputFile << ", ";
        }
        outputFile << endl;
    } 
}
void drawing_by_python(vector<layer_t> &nn, int depth) {
    vvd data;
    for (int i=0; i<depth; ++i) {
        nn[i].b = expansion_bias(nn[i].b, 1);
    }
    for (double x=-10; x<=10; x+=0.5) {
        for (double y=-10; y<=10; y+=0.5) {
            vvd tmp = {{x, y}};
            //forward propagation
            for (int k=0; k<depth; ++k) {
                if (k == 0) nn[k].a = matrix_add(matrix_multi(tmp, nn[k].w), nn[k].b);
                else nn[k].a = matrix_add(matrix_multi(nn[k-1].x, nn[k].w), nn[k].b);
                // if (k < depth-1) nn[k].x = hm_ReLU(nn[k].a);
                if (k < depth-1) nn[k].x = hm_tanh(nn[k].a);
                else nn[k].x = hm_identity(nn[k].a);
                
            }
            tmp[0].push_back(nn[depth-1].x[0][0]);
            data.push_back(tmp[0]);
        }
    }
    outputfile(data);
}

//  ##   ##    ##     ######   ######    ####    ##  ##
//  ### ###   ####    # ## #    ##  ##    ##     ##  ##
//  #######  ##  ##     ##      ##  ##    ##      ####
//  #######  ##  ##     ##      #####     ##       ##
//  ## # ##  ######     ##      ## ##     ##      ####
//  ##   ##  ##  ##     ##      ##  ##    ##     ##  ##
//  ##   ##  ##  ##    ####    #### ##   ####    ##  ##

// show
void matrix_show(vvd &a) {
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
// biasは各行同じものがバッチサイズ分あるので最初の1行だけ表示
void matrix_show_b(vvd &a) {
    int m = a[0].size();
    for (int j=0; j<m; ++j) cout << "--";
    cout << endl;

    for (int j=0; j<m; ++j) cout << a[0][j] << ' ';
    cout << endl;

    for (int j=0; j<m; ++j) cout << "--";
    cout << endl;
}

// c = a * b
vvd matrix_multi(const vvd &a, const vvd &b) {
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
vvd matrix_adm_multi(const vvd &a, const vvd &b) {
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
vvd matrix_add(const vvd &a, const vvd &b) {
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
vvd matrix_t(const vvd &a) {
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
// c = a .* b, a:matrix, b:tensor
vvd matrix_adm_multi_tensor(const vvd &a, const vvvd &b) {
    if (a.size() != b.size()) {
        cout << "The matrix sizes are different. 1dimention" << endl;
        return {{}};
    }
    if (a[0].size() != b[0].size()) {
        cout << "The matrix sizes are different. 2dimention" << endl;
    }
    int n = a.size(), m = a[0].size();
    vvd ret(n, vd(m, 0));
    for (int i=0; i<n; ++i) {
        vvd tmp = {a[i]};
        tmp = matrix_multi(tmp, b[i]);
        ret[i] = tmp[0];
    }
    return ret;
}

//    ##       ####   ######    ####    ##   ##    ##     ######    ####     #####   ##   ##
//   ####     ##  ##  # ## #     ##     ##   ##   ####    # ## #     ##     ##   ##  ###  ##
//  ##  ##   ##         ##       ##      ## ##   ##  ##     ##       ##     ##   ##  #### ##
//  ##  ##   ##         ##       ##      ## ##   ##  ##     ##       ##     ##   ##  ## ####
//  ######   ##         ##       ##       ###    ######     ##       ##     ##   ##  ##  ###
//  ##  ##    ##  ##    ##       ##       ###    ##  ##     ##       ##     ##   ##  ##   ##
//  ##  ##     ####    ####     ####       #     ##  ##    ####     ####     #####   ##   ##

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
double h_identity(double x) {
    return x;
}
double h_tanh(double x) {
    return (exp(x)-exp(-x)) / (exp(x)+exp(-x));
}
vvd hm_ReLU(vvd &x) {
    int n = x.size(), m = x[0].size();
    vvd tmp(n, vd(m));
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            tmp[i][j] = h_ReLU(x[i][j]);
        }
    }
    return tmp;
}

vvd hm_softmax(vvd &x) {
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
double hm_cross_entropy(vvd &y, vvd &t) {
    int n = y.size(), m = y[0].size();
    double sum = 0;
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            if (y[i][j] <= 0) y[i][j] = 1e-5;
            if (t[i][j]) sum += t[i][j] * log(y[i][j]);
        }
    }
    return -sum/n;
}
vvd hm_identity(vvd &x) {
    return x;
}
double hm_MSE(vvd &y, vvd &t) {
    int n = y.size(), m = y[0].size();
    if (m != 1) {
        cout << "not 1value regression" << endl;
        return 0;
    }
    double sum = 0;
    for (int i=0; i<n; ++i) {
        sum += pow(y[i][0] - t[i][0], 2);
    }
    return sum / (2*n);
}
vvd hm_tanh(vvd &x) {
    int n = x.size(), m = x[0].size();
    vvd tmp(n, vd(m));
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            tmp[i][j] = h_tanh(x[i][j]);
        }
    }
    return tmp;
}

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

vvd expansion_bias(vvd &b, int batch) {
    vvd c;
    for (int i=0; i<batch; ++i) {
        c.push_back(b[0]);
    }
    return c;
}

vvd calc_r_cross_entropy(vvd &x, vvd &t) {
    int n = x.size(), m = x[0].size();
    vvd tmp(n, vd(m, 0));
    for (int s=0; s<n; ++s) {
        for (int j=0; j<m; ++j) {
            for (int k=0; k<m; ++k) {
                if (j == k) tmp[s][j] -= t[s][j] / x[s][j];
                else tmp[s][j] += t[s][k] / x[s][k];
            }
            tmp[s][j] /= n;
        }
    }
    return tmp;
}

//rx_k/ra_j
//m class 分類
//m次正方行列を返す
vvvd calc_r_softmax(vvd &x) {
    int n = x.size(), m = x[0].size();
    vvvd ret(n, vvd(m, vd(m, 0)));
    for (int s=0; s<n; ++s) {
        for (int i=0; i<m; ++i) {
            for (int j=0; j<m; ++j) {
                if (i == j) ret[s][i][j] = x[s][i]*(1 - x[s][j]);
                else ret[s][i][j] = x[s][i]*(0 - x[s][j]);
            }
        }
    }
    return ret;
}

vvd calc_r_ReLU (vvd &a) {
    int n = a.size(), m = a[0].size();
    vvd tmp(n, vd(m, 0));
    for (int s=0; s<n; ++s) {
        for (int j=0; j<m; ++j) {
            if (a[s][j] >= 0) tmp[s][j] = 1;
        }
    }
    return tmp;
}

vvd calc_r_bias (vvd &b, vvd &delta) {
    int n = b.size(), m = b[0].size();
    vvd rb;
    if (n != delta.size() || m != delta[0].size()) cout << "size is not match" << endl;
    rb.assign(1, vd(m, 0));
    for (int j=0; j<m; ++j) {
        for (int i=0; i<n; ++i) {
            rb[0][j] += delta[i][j];
        }
    }
    rb = expansion_bias(rb, n);
    return rb;
}
vvd calc_r_identity(vvd &x) {
    int n = x.size(), m = x[0].size();
    vvd tmp(n, vd(m, 1));
    return tmp;
}
vvd calc_r_MSE(vvd &y, vvd &t) {
    int n = y.size(), m = y[0].size();
    vvd tmp(n, vd(m, 0));
    if (m != 1) {
        cout << "not 1value regression" << endl;
        return {{}};
    }
    for (int i=0; i<n; ++i) {
        tmp[i][0] = (y[i][0] - t[i][0]) / n;
    }
    return tmp;
}
vvd calc_r_tanh(vvd &a) {
    int n = a.size(), m = a[0].size();
    vvd tmp(n, vd(m, 0));
    for (int s=0; s<n; ++s) {
        for (int j=0; j<m; ++j) {
            tmp[s][j] = 4/(exp(-a[s][j]) + exp(a[s][j])) / (exp(-a[s][j]) + exp(a[s][j]));
        }
    }
    return tmp;
}

void updateWeights(vvd &w, vvd &rw, double eta) {
    if (!(w.size() == rw.size() && w[0].size() == rw[0].size())) {
        cout << "The matrix sizes are different." << endl;
        cout << "in update weight" << endl;
        cout << w.size() << ' ' << w[0].size() << endl;
        cout << rw.size() << ' ' << rw[0].size() << endl;
    }
    int n = w.size(), m = w[0].size();
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            w[i][j] -= eta * rw[i][j];
        }
    }
}
