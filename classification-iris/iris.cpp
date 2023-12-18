/*
iris datasets
*/
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <set>
#include <math.h>
#include <string>
#include <fstream>
#include <sstream>
#include <random>


using namespace std;

#define pi 3.14159265358979323846
#define yes "Yes"
#define no "No"
#define yesno(bool) if(bool){cout<<"Yes"<<endl;}else{cout<<"No"<<endl;}
#define tp() cout << "here~~" << endl

using vd = vector<double>;
using vvd = vector<vector<double>>;

random_device rd;
long long seed = 0;//rd()
// long long seed = rd();
mt19937 engine(seed);

double gaussianDistribution (double mu, double sig) {
    normal_distribution <> dist(mu, sig);
    return dist(engine);
}
//rd()は実行毎に異なる
//rand()は実行毎に同じ
mt19937 gen(seed);//seed
uniform_real_distribution<> distCircle(-6, 6);

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
// bool judgeTerm(double x, double y){ return (y > x) ? true : false;}
void makeData(vvd &x, int n, int seed=0) {
    //条件を満たす点と満たさない点をｎ個ずつ作る
    
    x.assign(2*n, vd(2, 0));
    int id = 0;
    while(id < n) {
        double a, b;
        a = distCircle(gen);
        b = distCircle(gen);
        if (judgeTerm(a, b)) {
            x[id][0] = a;
            x[id][1] = b;
            ++id;
        }
    }
    while(id < n*2) {
        double a, b;
        a = distCircle(gen);
        b = distCircle(gen);
        if (!judgeTerm(a, b)) {
            x[id][0] = a;
            x[id][1] = b;
            ++id;
        }
    }
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
#include <fstream>
void outputfile(vvd &x) {
    int n = x.size(), m = x[0].size();
    string fname1 = "in.txt";
    string fname2 = "out.txt";
    ofstream outputFile (fname1);
    ofstream outputFile2 (fname2);
    

    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            if (i < n/2) {
                outputFile << x[i][j];
                if (j != m-1) outputFile << " ";
            } else {
                outputFile2 << x[i][j];
                if (j != m-1) outputFile2 << " ";
            }
            
        }
        if (i < n/2) {
            outputFile << endl;
        } else {
            outputFile2 << endl;
        }
        
    } 
}
void outputTextFile2d(vvd &v, string s) {  
}


vvd readCSV(string filename) {
    ifstream file(filename);
    vvd data;
    if (!file.is_open()) {
        cerr << "Error Opening File!" << endl;
        return data;
    }

    string line;
    while (getline(file, line)) {
        vd row;
        stringstream ss(line);
        string cell;
        while (getline(ss, cell, ',')) {
            row.push_back(stod(cell));
        }
        data.push_back(row);
    }
    file.close();
    return data;
}




/*
all 150
train 105:35, 35, 35
test 45:15, 15, 15

train 93:31, 31, 31
valid 12:4, 4, 4
test 45:15, 15, 15
50, 50, 50
*/
int main() {
    //data import
    ////////////////////////////////////////////////////////////////
    vvd data = readCSV("data/iris_datasets_data.csv");
    for (const auto& row : data) {
        for (const auto& cell : row) {
            cout << cell << " ";
        }
        cout << endl;
    }
    cout << data.size() << ' ' << data[0].size() << endl;
    
    vvd target_tmp = readCSV("data/iris_datasets_target.csv");
    vvd target;
    for (int i=0; i<target_tmp[0].size(); ++i) {
        if (target_tmp[0][i] == 0) target.push_back({1, 0, 0});
        else if (target_tmp[0][i] == 1) target.push_back({0, 1, 0});
        else if (target_tmp[0][i] == 2) target.push_back({0, 0, 1});
    }

    for (const auto& row : target) {
        for (const auto& cell : row) {
            cout << cell << " ";
        }
        cout << endl;
    }
    cout << target.size() << ' ' << target[0].size() << endl;
    ////////////////////////////////////////////////////////////////
    return 0;

    vvd x0, x1, x2, x3, a1, a2, a3, w1, w2, w3, b1, b2, b3;
    vvd tmp1, r_hL_x3, r_h3_a3, Delta3, r_L_w3, tx2, r_h2_a2, tw3, tmp2, Delta2, tx1, r_L_w2, r_h1_a1, tw2, tmp3, Delta1, tx0, r_L_w1, r_L_b1, r_L_b2, r_L_b3;
    double eta = 0.1;
    int n = 1000;

    vector<int> id(n);
    for (int i=0; i<n; ++i) id[i] = i;
    shuffle(id.begin(), id.end(), engine);
    
    w1.assign(2, vd(3, 0));
    w2.assign(3, vd(3, 0));
    w3.assign(3, vd(2, 0));
    makeInitialValue(w1, 0, sqrt(2.0/2));
    makeInitialValue(w2, 0, sqrt(2.0/3));
    makeInitialValue(w3, 0, sqrt(2.0/3));
    b1.assign(1, vd(3, 0));
    b2.assign(1, vd(3, 0));
    b3.assign(1, vd(2, 0));
    makeInitialValue(b1, 0, sqrt(2.0/3));
    makeInitialValue(b2, 0, sqrt(2.0/3));
    makeInitialValue(b3, 0, sqrt(2.0/2));
    // expansionBias(b1, n);
    // expansionBias(b2, n);
    // expansionBias(b3, n);
    expansionBias(b1, 1);
    expansionBias(b2, 1);
    expansionBias(b3, 1);
    
    cout << "w" << endl;
    showMatrix(w1);
    showMatrix(w2);
    showMatrix(w3);

    // cout << "b" << endl;
    // showMatrix(b1);
    // showMatrix(b2);
    // showMatrix(b3);
    
    // return 0;

    //教師データの作成 {1,0}内側
    vvd t;
    vd tmp = {1, 0};//inner
    for (int i=0; i<n/2; ++i) t.push_back(tmp);
    tmp = {0, 1};//outer
    for (int i=0; i<n/2; ++i) t.push_back(tmp);
    
    makeData(x0, n/2);



    //learn
    for (int i=0; i<1500; ++i) {
        if (i % 800 == 0) eta *= 0.5;
        //forward propagation
        shuffle(id.begin(), id.end(), engine);
        shuffleVVD(t, id);
        shuffleVVD(x0, id);
        // cout << "first x0" << endl;
        // showMatrix(x0);
        // cout << "t" << endl;
        // showMatrix(t);
        vvd x00 = {{x0[0][0], x0[0][1]}};
        vvd t00 = {{t[0][0], t[0][1]}};
        
        //a1 = w1 * x0 + b1
        multiMatrix(tmp1, x00, w1);
        addMatrix(a1, tmp1, b1);
        x1 = h_ReLUMatrix(a1);
        //a2 = w2 * x1 + b2
        multiMatrix(tmp1, x1, w2);
        addMatrix(a2, tmp1, b2);
        x2 = h_ReLUMatrix(a2);
        //a3 = w3 * x2 + b3
        multiMatrix(tmp1, x2, w3);
        addMatrix(a3, tmp1, b3);
        x3 = softMax(a3);
        if (i % 500 == 0) {
            cout << i << " cross entropy ";
            cout << crossEntropy(x3, t00) << endl;
            cout << "accuracy rate ";
            cout << calcAccuracyRate(x3, t00) << endl;
        }

        //back propagation
        r_hL_x3 = calc_r_hL_x3(x3, t);
        r_h3_a3 = calc_r_h3_a3(a3, x3);
        admMultiMatrix(Delta3, r_hL_x3, r_h3_a3);
        calc_r_L_b(r_L_b3, b3, Delta3);

        tx2 = x2;
        tMatrix(tx2);
        multiMatrix(r_L_w3, tx2, Delta3);
        r_h2_a2 = calc_r_h2_a2(a2, x2);
        tw3 = w3;
        tMatrix(tw3);
        multiMatrix(tmp2, Delta3, tw3);
        admMultiMatrix(Delta2, r_h2_a2, tmp2);
        calc_r_L_b(r_L_b2, b2, Delta2);

        tx1 = x1;
        tMatrix(tx1);
        multiMatrix(r_L_w2, tx1, Delta2);
        r_h1_a1 = calc_r_h2_a2(a1, x1);
        tw2 = w2;
        tMatrix(tw2);
        multiMatrix(tmp3, Delta2, tw2);
        admMultiMatrix(Delta1, r_h1_a1, tmp3);
        calc_r_L_b(r_L_b1, b1, Delta1);

        tx0 = x0;
        tMatrix(tx0);
        multiMatrix(r_L_w1, tx0, Delta1);

        updateWeights(w1, r_L_w1, eta);
        updateWeights(w2, r_L_w2, eta);
        updateWeights(w3, r_L_w3, eta);
        updateWeights(b1, r_L_b1, eta);
        updateWeights(b2, r_L_b2, eta);
        updateWeights(b3, r_L_b3, eta);
        
    }

    // train set---------------------------------
    cout << "=========================================" << endl;
    cout << "train set rate" << endl;
    expansionBias(b1, n);
    expansionBias(b2, n);
    expansionBias(b3, n);
    // t.assign(0, vd(0));
    multiMatrix(tmp1, x0, w1);
    addMatrix(a1, tmp1, b1);
    x1 = h_ReLUMatrix(a1);
    //a2 = w2 * x1 + b2
    multiMatrix(tmp1, x1, w2);
    addMatrix(a2, tmp1, b2);
    x2 = h_ReLUMatrix(a2);
    //a3 = w3 * x2 + b3
    multiMatrix(tmp1, x2, w3);
    addMatrix(a3, tmp1, b3);
    x3 = softMax(a3);

    cout << "cross entropy";
    cout << crossEntropy(x3, t) << endl;
    cout << "accuracy rate ";
    cout << calcAccuracyRate(x3, t) << endl;

    // test set-------------------------------------
    cout << "=========================================" << endl;
    cout << "test" << endl;
    
    makeData(x0, n/2);
    
    t.assign(0, vd(0));
    tmp = {1, 0};//inner
    for (int i=0; i<n/2; ++i) t.push_back(tmp);
    tmp = {0, 1};//outer
    for (int i=0; i<n/2; ++i) t.push_back(tmp);

    // cout << "test answer" << endl;
    // showMatrix(t);

    // cout << "test x0" << endl;
    // showMatrix(x0);

    //a1 = w1 * x0 + b1
    multiMatrix(tmp1, x0, w1);
    addMatrix(a1, tmp1, b1);
    x1 = h_ReLUMatrix(a1);
    //a2 = w2 * x1 + b2
    multiMatrix(tmp1, x1, w2);
    addMatrix(a2, tmp1, b2);
    x2 = h_ReLUMatrix(a2);
    //a3 = w3 * x2 + b3
    multiMatrix(tmp1, x2, w3);
    addMatrix(a3, tmp1, b3);
    x3 = softMax(a3);

    test = true;

    cout << "cross entropy";
    cout << crossEntropy(x3, t) << endl;
    cout << "accuracy rate ";
    cout << calcAccuracyRate(x3, t) << endl;

    cout << "=========================================" << endl;
    showMatrix(w1);
    showMatrix(w2);
    showMatrix(w3);
    showMatrixB(b1);
    showMatrixB(b2);
    showMatrixB(b3);
    cout << "=========================================" << endl;
    
    
    

}


/*


*/