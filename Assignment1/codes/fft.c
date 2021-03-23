#include <stdio.h>
#include <complex.h>
#include <math.h>

int fft(int n, complex v[n], complex temp[n])
{
    if (n > 1)
    {
        for (int k = 0; k < n / 2; k++)
        {
            temp[k + n / 2] = v[2 * k];
            temp[k] = v[2 * k + 1];
        }
        fft(n / 2, temp, v);
        fft(n / 2, temp + n / 2, v);
        for (int m = 0; m < n / 2; m++)
        {
            complex w = cos(2 * M_PI * m / (double)n) - (sin(2 * M_PI * m / (double)n)) * I;
            v[m] = temp[m + n / 2] + w * temp[m];
            v[m + n / 2] = temp[m + n / 2] - w * temp[m];
        }
    }
    return 0;
}
int main()
{
    int N = 8;
    complex v[N], temp[N];
    float x[] = {1, 2, 3, 4, 2, 1, 0, 0};
    for (int k = 0; k < N; k++)
        v[k] = x[k] + 0 * I;
    fft(N, v, temp);
    for (int i = 0; i < N; i++)
        printf("(%.5lf, %.5lf)\n", creal(v[i]), cimag(v[i]));
}