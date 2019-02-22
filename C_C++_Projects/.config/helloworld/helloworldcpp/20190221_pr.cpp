#include <stdio.h>

int add_number(int *parr);
int main(){
    int arr[3];
    int i;

    for(i = 0; i<3; i++){
        scanf("%d", &arr[i]);
    }

    add_number(arr);

    printf("배열의 각 원소 : %d, %d, %d", arr[0], arr[1], arr[2]);

    return 0;
}

int add_number(int *parr){
    int i;
    for(i = 0; i<3; i++){
        parr[i]++;
    }
    return 0;
}