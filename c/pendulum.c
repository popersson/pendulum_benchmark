#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

void fpend(double *y, double *f)
{
    double th1, th2, om1, om2, s1, c1, s2, c2, sd, cd, s12, denom;
    
    th1 = y[0];
    th2 = y[1];
    om1 = y[2];
    om2 = y[3];

    s1 = sin(th1);          /* each sin/cos pair costs one sincos call */
    c1 = cos(th1);
    s2 = sin(th2);
    c2 = cos(th2);
    sd = s1*c2 - c1*s2;     /* sin(th1-th2)        */
    cd = c1*c2 + s1*s2;     /* cos(th1-th2)        */
    s12 = sd*c2 - cd*s2;    /* sin(th1-2*th2)      */
    denom = 2 + 2*sd*sd;    /* 3-cos(2*th1-2*th2)  */

    f[0] = om1;
    f[1] = om2;
    f[2] = (-3*s1 - s12 - 2*sd*(om2*om2 + om1*om1*cd)) / denom;
    f[3] = 2*sd*(2*om1*om1 + 2*c1 + om2*om2*cd) / denom;
}

#define y(i,j) y[(i) + 4*(j)]

#define copy4(x,y) { (y)[0] = (x)[0]; (y)[1] = (x)[1]; (y)[2] = (x)[2]; (y)[3] = (x)[3]; }

void runge5(double *y0, double h, int N, double *y)
{
    double k1[4], k2[4], k3[4], k4[4], k5[4], k6[4], ytmp[4];

    for (int i=0; i<4; i++)
        y(i,0) = y0[i];

    for (int n=0; n<N; n++) {
        copy4(&y(0,n), ytmp);
        fpend(ytmp, k1);

        copy4(&y(0,n), ytmp);
        for (int i=0; i<4; i++)
            ytmp[i] += h*k1[i]/5;
        fpend(ytmp, k2);
        
        copy4(&y(0,n), ytmp);
        for (int i=0; i<4; i++)
            ytmp[i] += h*2*k2[i]/5;
        fpend(ytmp, k3);
        
        copy4(&y(0,n), ytmp);
        for (int i=0; i<4; i++)
            ytmp[i] += h*9*k1[i]/4 - h*5*k2[i] + h*15*k3[i]/4;
        fpend(ytmp, k4);

        copy4(&y(0,n), ytmp);
        for (int i=0; i<4; i++)
            ytmp[i] += -h*63*k1[i]/100 + h*9*k2[i]/5 - h*13*k3[i]/20 + h*2*k4[i]/25;
        fpend(ytmp, k5);

        copy4(&y(0,n), ytmp);
        for (int i=0; i<4; i++)
            ytmp[i] += -h*6*k1[i]/25 + h*4*k2[i]/5 + h*2*k3[i]/15 + h*8*k4[i]/75;
        fpend(ytmp, k6);

        for (int i=0; i<4; i++)
            y(i,n+1) = y(i,n) + h*(17*k1[i] + 100*k3[i] + 2*k4[i] - 50*k5[i] + 75*k6[i]) / 144;
    }
}

int main()
{
    double y0[4] = {2.0, 2.0, 0.0, -1.0};
    double h = 0.2;
    double T = 10000.0;

    int N = round(T/h);
    double *y;
    clock_t time_elapsed;

    for (int iter=0; iter<10; iter++) {
        time_elapsed = clock();
        y = calloc(4*(N+1), sizeof(double));
        runge5(y0, h, N, y);
        free(y);
        time_elapsed = clock() - time_elapsed;
        printf("Time %f seconds.\n", (float)time_elapsed/CLOCKS_PER_SEC);
    }

    return 0;
}

