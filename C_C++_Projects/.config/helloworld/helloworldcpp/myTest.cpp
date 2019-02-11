#include <iostream>
using namespace std;

int chage_val(int *p){
    *p = 3;

    return 0;
}

int main(){
    int number = 5;

    cout << number << endl;
}