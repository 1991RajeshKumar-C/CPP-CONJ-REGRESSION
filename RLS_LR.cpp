#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
// #include <sample>
using namespace std;


// Excepted output: [105.21455835 142.67095131 132.93605469 129.70175405]
// int iter = 2;

void gradientsdisp(float* grad, int F)
{
    for (int i = 0; i <= F; i++)
        cout << grad[i] << " ";
    cout << endl;
}

float* CGregression(float** arr, int F, int N)
{
    // C: float* arrloc = new float[F+1];
    // A.T @ A computation, SIZE: F x F
    float** MAT = new float*[F+1];
    for(int i = 0; i <= F; i++)
    {
        MAT[i] = new float[F+1];
        for(int j = 0; j <= F; j++)
        {
            if (i < F && j < F)
            {
                float dot_prod_temp = 0.0;
                for (int k = 0; k < N; k++)
                    dot_prod_temp += arr[k][i] * arr[k][j];
                MAT[i][j] = dot_prod_temp;
            }
            else if (i < F)
            {
                float dot_prod_temp = 0.0;
                for (int k = 0; k < N; k++)
                    dot_prod_temp += arr[k][i];
                MAT[i][j] = dot_prod_temp;
            }
            else if (j < F)
            {
                float dot_prod_temp = 0.0;
                for (int k = 0; k < N; k++)
                    dot_prod_temp += arr[k][j];
                MAT[i][j] = dot_prod_temp;
            }
            else{
                MAT[i][j] = (float)N;
            }
        }
    }

    // A.T @ b computation, SIZE: F x 1
    float* VEC = new float[F+1];
    for(int i = 0; i <= F; i++)
    {
        if (i < F)
        {
            float dot_prod_temp = 0.0;
            for (int k = 0; k < N; k++)
                dot_prod_temp += arr[k][i] * arr[k][F];
            VEC[i] = dot_prod_temp;
        }
        else{
            float dot_prod_temp = 0.0;
            for (int k = 0; k < N; k++)
                dot_prod_temp += arr[k][F];
            VEC[F] = dot_prod_temp;
        }
    }

    // float* arrloc = new float[F+1];
    float* arrloc = new float[F+1];   // without bias
    float* grad_old = new float[F+1];
    float* grad_oldv1 = new float[F+1];

    for (int i = 0; i < F; i++)
    {
        arrloc[i] = 0.0;
    }
    arrloc[F] = 0.0;

    // float loss = 0.0;       
    for (int lo = 0; lo <= F; lo++)
    {
        float dot_prod_temp = 0.0;
        for (int i = 0; i <= F; i++)
            dot_prod_temp += arrloc[i] * MAT[lo][i];
        // dot_prod_temp += arrloc[F];
        float res = VEC[lo] - dot_prod_temp;
        grad_old[lo] = res;
        grad_oldv1[lo] = grad_old[lo];
    }
         
    for (int k = 0; k <= F; k++)
    {
        float lr = 0.0, lrv1 = 0.0; 
        // float dot_prod = 0.0;
        // for (int i = 0; i < F; i++)
        //     dot_prod += arrloc[i] * arr[k][i];
        // dot_prod += arrloc[F];
        // float res = arr[k][F] - dot_prod;

        // loss += res * res;

        float num = 0.0;
        float den = 0.0;

        // dot_prod = 0.0;
        for (int i = 0; i <= F; i++)
        {
            num += (grad_old[i] * grad_old[i]);
            for (int j = 0; j <= F; j++)
                den += (MAT[i][j] * grad_oldv1[i] * grad_oldv1[j]);
        }
        if (den >= 0.00001 || den <= -0.00001)
            // Learning rate
            lr = num / den;

        // float res = arr[k][F] - dot_prod;
        // dot_prod = den;
        num = 0.0;
        den = 0.0;
        for (int i = 0; i <= F; i++)
        {
            float dot_prod = 0.0;
            den += (grad_old[i] * grad_old[i]);
            arrloc[i] += (lr * grad_oldv1[i]);
            cout << "Unknown: " << arrloc[i] << ", " << endl;
            // cout << "Grad[" << i << "]: " << grad_oldv1[i] << ", " << lr * grad_oldv1[i] << endl;
            // cout << "Grad_Before: " << grad_old[i] << ", " << lr * dot_prod << ", " << lr * dot_prod * arr[k][i] << endl;
            for (int j = 0; j <= F; j++)
                dot_prod += (MAT[i][j] * grad_oldv1[j]);
            grad_old[i] -= lr * dot_prod;       // check later for sign

            // cout << "Grad1: " << grad_old[i] << ", " << endl;
            num += (grad_old[i] * grad_old[i]); 
        }
        // arrloc[F] += (lr * res/(float)N);
        // cout << "Unknown: " << arrloc[F] << ", " << endl;
        if (den >= 0.00001 || den <= -0.00001)
                lrv1 = num / den;
        for (int i = 0; i <= F; i++)
        {
            grad_oldv1[i] = grad_old[i] + lrv1 * grad_oldv1[i];
            // cout << "Grad2: " << grad_oldv1[i] << ", " << endl;
        }
        // cout << "lr: " << lr << ", " << "lr1; " << lrv1 << endl;
        gradientsdisp(grad_oldv1, F);
        cout << "=======================================" << endl;
        // gradientsdisp(grad_old, F);
        // C: arrloc[F] -= (lr * -res) / (float)N;
    }
    return arrloc;
}


int main() {
    /* Enter your code here. Read input from STDIN. Print output to STDOUT */  
    fstream FILE;
    int F, N, T;
    FILE.open("sample", ios::in);
    

    if (FILE.is_open())
    {
        string line;
        getline(FILE, line);
        stringstream ss(line);
        ss >> F >> N;
        cout << F << " " << N << endl;
        
        float ** arr = new float*[N];
        for (int i = 0; i < N; i++)
        {
            arr[i] = new float[F+1];
            string line;
            getline(FILE, line);
            stringstream ss(line);
            for (int j = 0; j <= F; j++)
            {
                ss >> arr[i][j];
                cout << arr[i][j] << " ";
            }
            cout << endl;
        }

        getline(FILE, line);
        stringstream ss1(line);
        ss1 >> T;
        cout << T << endl;
        float** arr1 = new float*[T];
        for (int i = 0; i < T; i++)
        {
            arr1[i] = new float[F];
            string line;
            getline(FILE, line);
            stringstream ss(line);
            for (int j = 0; j < F; j++)
            {
                ss >> arr1[i][j];
                cout << arr1[i][j] << " ";
            }
            cout << endl;
        }

        FILE.close();

        float* coeff = CGregression(arr, F, N);
        cout << "Fit parameters" << endl;
        for (int i = 0; i <= F; i++)
            cout << coeff[i] << " ";
        cout << endl;

        for (int i = 0; i < T; i++)
        {
            float dot_prod = 0.0;
            for (int j = 0; j < F; j++)
            {
                dot_prod += coeff[j] * arr1[i][j];
            }
            dot_prod += coeff[F];
            cout << dot_prod << endl;
        }
    }
    return 0;
}
