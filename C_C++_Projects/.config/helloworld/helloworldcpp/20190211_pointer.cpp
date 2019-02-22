#include <stdio.h>
#include <iostream>

using namespace std;

int change_val(int *pi){
    cout << "chnage 함수 내부" << endl;
    printf("pi의 값 : %d \n", pi);
    printf("pi가 가리키는 것의값 %d \n");

    *pi = 3;

    printf("----change val 함수 끝");
    return 0;
}

int main(){
    int i = 0;

    printf("i 변수의 주소값 : %d \n ", &i);
    printf("호출 이전 i의 값 : %d \n", i);
    change_val(&i);
    printf("호출 이후 i의 값 : %d \n", i);

    return 0;
}